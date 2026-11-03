//! Command line: `<exe> fem <command>` for scripts and sweeps without the window.
//!   fem example-sweep              print a sweep definition (JSON) to start from
//!   fem write <sweep.json> [-w dir] write the job bundles of a sweep
//!   fem run [-w dir]               run every bundle without results (RAM/seat-limited queue)
//!   fem compare [-w dir] [--tol x] effective tensors per case and engine, differences, flags

use super::adapters::{Engine, EngineSettings};
use super::compare;
use super::jobs::{self, Machine, Range, Status, Sweep};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub fn main(args: &[String]) -> i32 {
    let mut ws = PathBuf::from("fem-workspace");
    let mut tol = 0.02;
    let mut pos = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--workspace" => {
                i += 1;
                ws = PathBuf::from(args.get(i).cloned().unwrap_or_default());
            }
            "--tol" => {
                i += 1;
                tol = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(tol);
            }
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    match pos.first().map(|s| s.as_str()) {
        Some("example-sweep") => {
            let s = Sweep {
                name: "theta_t_grid".into(),
                base: super::default_job(),
                theta_deg: Range { from: 5.0, to: 25.0, steps: 5 },
                t: Range { from: 0.6, to: 1.4, steps: 3 },
                engines: vec![Engine::CodeAster, Engine::Fenicsx],
            };
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            0
        }
        Some("write") => {
            let Some(path) = pos.get(1) else {
                eprintln!("usage: fem write <sweep.json> [-w workspace]");
                return 1;
            };
            let sweep: Sweep = match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return 1;
                }
            };
            let ex = sweep.expand();
            let settings = EngineSettings::default();
            let mut n = 0;
            for j in &ex.jobs {
                for &e in &sweep.engines {
                    match jobs::write_bundle(&ws, &sweep.name, j, e, &settings) {
                        Ok(b) => {
                            println!("{}", b.dir.display());
                            n += 1;
                        }
                        Err(err) => {
                            eprintln!("{err}");
                            return 1;
                        }
                    }
                }
            }
            for (k, why) in &ex.rejected {
                eprintln!("rejected {k}: {}", why.join("; "));
            }
            eprintln!("{n} bundle(s) written, {} parameter set(s) rejected", ex.rejected.len());
            0
        }
        Some("run") => {
            let todo: Vec<_> = jobs::scan(&ws).into_iter().filter(|b| b.status != Status::Done).collect();
            let p = Arc::new(Mutex::new(Vec::new()));
            let n = todo.len();
            jobs::run_all(todo, Machine::default(), p.clone());
            let failed = p.lock().unwrap().iter().filter(|x| matches!(x.1, Status::Failed(_))).count();
            eprintln!("{n} bundle(s) run, {failed} failed");
            if failed > 0 { 2 } else { 0 }
        }
        Some("export") => {
            // fem export <code_aster|fenicsx|mapdl> <code.dat> [-w dir]: write the engine job for the code
            let engine = Engine::ALL.into_iter().find(|e| Some(e.key()) == pos.get(1).map(|s| s.as_str()));
            let (Some(engine), Some(path)) = (engine, pos.get(2)) else {
                eprintln!("usage: fem export <code_aster|fenicsx|mapdl> <code.dat> [-w workspace]");
                return 1;
            };
            let code = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return 1;
                }
            };
            match super::export::prepare(&code, engine, &EngineSettings::default(), &ws) {
                Ok(d) => {
                    println!("{}", d.display());
                    0
                }
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        Some("compare") => {
            let all = jobs::scan(&ws);
            let mut cases: Vec<String> = all.iter().map(|b| b.case_id.clone()).collect();
            cases.dedup();
            let mut report = Vec::new();
            let mut flagged = 0;
            for c in cases {
                let effs: Vec<_> = all.iter().filter(|b| b.case_id == c).filter_map(compare::effective).collect();
                if effs.is_empty() {
                    continue;
                }
                let rows = compare::cross_check(&effs, tol);
                println!("case {c}");
                for r in &rows {
                    let vals: Vec<String> = r.values.iter().map(|(e, v)| format!("{}={v:.6e}", e.key())).collect();
                    println!("  {:<5} {}  diff {:.3}%{}", r.name, vals.join("  "), r.max_rel_diff * 100., if r.flagged { "  FLAG" } else { "" });
                    if r.flagged {
                        flagged += 1;
                    }
                }
                report.push(serde_json::json!({
                    "case_id": c,
                    "engines": effs,
                    "rows": rows.iter().map(|r| serde_json::json!({"name": r.name, "max_rel_diff": r.max_rel_diff, "flagged": r.flagged,
                        "values": r.values.iter().map(|(e, v)| (e.key().to_string(), *v)).collect::<std::collections::BTreeMap<_, _>>()})).collect::<Vec<_>>(),
                }));
            }
            let _ = std::fs::write(ws.join("comparison.json"), serde_json::to_string_pretty(&report).unwrap_or_default());
            if flagged > 0 { 3 } else { 0 }
        }
        _ => {
            eprintln!("fem commands: example-sweep | write <sweep.json> | run | compare   (options: -w <workspace>, --tol <fraction>)");
            1
        }
    }
}
