//! The rebuilt (reverse-engineered) MAPDL-compatible solver used as an engine of the neutral model.
//! It runs in-process: shared Gmsh mesh -> APDL deck (N, E, CE, D, SOLVE) -> this program's own
//! interpreter and solver -> effective stress from the constraint forces (Hill's lemma).

use crate::apdl::Session;
use crate::solver::{Settings, SolverKind};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

pub struct Mesh {
    pub nodes: HashMap<usize, [f64; 2]>,
    /// triangles: 3 corner nodes, or 3 corners + 3 mid-side nodes (Gmsh order)
    pub tris: Vec<Vec<usize>>,
}

/// minimal reader for Gmsh MSH 4.1 ASCII: nodes and 2-D triangles (types 2 and 9)
pub fn read_msh(path: &Path) -> Result<Mesh, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = txt.lines();
    let mut nodes = HashMap::new();
    let mut tris = Vec::new();
    let next = |it: &mut std::str::Lines| -> Result<Vec<f64>, String> {
        let l = it.next().ok_or("unexpected end of the mesh file")?;
        l.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| format!("bad number '{w}' in the mesh file"))).collect()
    };
    while let Some(l) = lines.next() {
        match l.trim() {
            "$MeshFormat" => {
                let v = lines.next().unwrap_or("");
                let w: Vec<&str> = v.split_whitespace().collect();
                if w.first().map(|s| !s.starts_with('4')).unwrap_or(true) || w.get(1) != Some(&"0") {
                    return Err(format!("only ASCII MSH 4.x is read (found '{v}')"));
                }
            }
            "$Nodes" => {
                let h = next(&mut lines)?;
                for _ in 0..h[0] as usize {
                    let b = next(&mut lines)?;
                    let (param, n) = (b[2] as usize, b[3] as usize);
                    let mut tags = Vec::with_capacity(n);
                    for _ in 0..n {
                        tags.push(next(&mut lines)?[0] as usize);
                    }
                    for t in tags {
                        let c = next(&mut lines)?;
                        let _ = param;
                        nodes.insert(t, [c[0], c[1]]);
                    }
                }
            }
            "$Elements" => {
                let h = next(&mut lines)?;
                for _ in 0..h[0] as usize {
                    let b = next(&mut lines)?;
                    let (dim, ty, n) = (b[0] as usize, b[2] as usize, b[3] as usize);
                    for _ in 0..n {
                        let v = next(&mut lines)?;
                        if dim == 2 && (ty == 2 || ty == 9) {
                            tris.push(v[1..].iter().map(|x| *x as usize).collect());
                        } else if dim == 2 {
                            return Err(format!("element type {ty} is not read by the rebuilt solver (triangles only)"));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err("the mesh has no triangles".into());
    }
    Ok(Mesh { nodes, tris })
}

fn geometry_step(dir: &Path, gmsh_pin: &str, log: &mut String) -> Result<(), String> {
    if dir.join("cell.msh").exists() && dir.join("geometry.json").exists() {
        return Ok(());
    }
    let tries: Vec<Vec<String>> = vec![
        vec!["uv".into(), "run".into(), "--no-project".into(), "--with".into(), gmsh_pin.into(), "python".into(), "geometry.py".into()],
        vec!["python".into(), "geometry.py".into()],
        vec!["python3".into(), "geometry.py".into()],
    ];
    for t in tries {
        if let Ok(o) = std::process::Command::new(&t[0]).args(&t[1..]).current_dir(dir).output() {
            let _ = writeln!(log, "$ {}\n{}{}", t.join(" "), String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
            if o.status.success() && dir.join("cell.msh").exists() {
                return Ok(());
            }
        }
    }
    Err("the Gmsh geometry step could not run (needs uv, or Python with the gmsh package)".into())
}

/// the APDL deck of one unit-cell bundle (also used by the Meshing tab to fill the code window)
pub struct Deck {
    pub text: String,
    pub cases: Vec<[f64; 3]>,
    pub l: f64,
    pub h: f64,
    pub npr: usize,
    pub npt: usize,
}

/// run the Gmsh geometry of a bundle and write its APDL deck (job_local.dat)
pub fn make_deck(dir: &Path, log: &mut String) -> Result<Deck, String> {
    let p: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("params.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let pin = std::fs::read_to_string(dir.join("bundle.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["pins"]["gmsh"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "gmsh".into());
    geometry_step(dir, &pin, log)?;
    let geo: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("geometry.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mesh = read_msh(&dir.join("cell.msh"))?;
    let f = |k: &str| p[k].as_f64().unwrap_or(0.);
    let (l, h, e, nu) = (f("L"), f("thickness"), f("E"), f("nu"));
    let cases: Vec<[f64; 3]> = serde_json::from_value(p["strain_cases"].clone()).map_err(|e| e.to_string())?;
    let quad = mesh.tris.first().map(|t| t.len() >= 6).unwrap_or(false);
    // ---- deck
    let mut d = String::with_capacity(64 * (mesh.nodes.len() + mesh.tris.len()));
    let _ = writeln!(d, "/TITLE, fem-client case (rebuilt MAPDL-compatible solver)\n/NOPR\n/PREP7");
    if quad {
        let _ = writeln!(d, "ET,1,PLANE183,1,,3        ! 6-node triangles, plane stress with thickness");
    } else {
        let _ = writeln!(d, "ET,1,PLANE182,,,3         ! 3-node triangles (K = L), plane stress with thickness");
    }
    let _ = writeln!(d, "R,1,{h}\nMP,EX,1,{e}\nMP,PRXY,1,{nu}");
    let mut ids: Vec<usize> = mesh.nodes.keys().copied().collect();
    ids.sort_unstable();
    for id in &ids {
        let x = mesh.nodes[id];
        let _ = writeln!(d, "N,{id},{:.15e},{:.15e}", x[0], x[1]);
    }
    let area = |t: &[usize]| {
        let (a, b, c) = (mesh.nodes[&t[0]], mesh.nodes[&t[1]], mesh.nodes[&t[2]]);
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    };
    for t in &mesh.tris {
        let t: Vec<usize> = if area(t) < 0. {
            if t.len() >= 6 { vec![t[0], t[2], t[1], t[5], t[4], t[3]] } else { vec![t[0], t[2], t[1]] }
        } else {
            t.clone()
        };
        if quad {
            let _ = writeln!(d, "E,{},{},{},{},{},{}", t[0], t[1], t[2], t[3], t[4], t[5]);
        } else {
            let _ = writeln!(d, "E,{},{},{},{}", t[0], t[1], t[2], t[2]);
        }
    }
    // ---- periodic pairs (found here, by this engine)
    let tol = 1e-6 * l;
    let near = |a: f64, b: f64| (a - b).abs() < tol;
    let side = |pred: &dyn Fn(&[f64; 2]) -> bool| -> Vec<usize> { ids.iter().copied().filter(|i| pred(&mesh.nodes[i])).collect() };
    let right = side(&|x| near(x[0], l / 2.));
    let left = side(&|x| near(x[0], -l / 2.));
    let top = side(&|x| near(x[1], l / 2.) && !near(x[0], l / 2.)); // top-right corner relation is implied
    let bottom = side(&|x| near(x[1], -l / 2.));
    let pair = |a: &[usize], b: &[usize], k: usize| -> Result<Vec<(usize, usize)>, String> {
        let o = 1 - k;
        a.iter()
            .map(|&s| {
                let xs = mesh.nodes[&s][o];
                b.iter()
                    .copied()
                    .min_by(|&p, &q| (mesh.nodes[&p][o] - xs).abs().partial_cmp(&(mesh.nodes[&q][o] - xs).abs()).unwrap())
                    .filter(|&m| (mesh.nodes[&m][o] - xs).abs() < 1e3 * tol)
                    .map(|m| (s, m))
                    .ok_or_else(|| format!("node {s} on a periodic face has no partner on the opposite face (the mesh is not periodic)"))
            })
            .collect()
    };
    let pr = pair(&right, &left, 0)?;
    let pt = pair(&top, &bottom, 1)?;
    let anc = geo["anchor"].as_array().map(|a| [a[0].as_f64().unwrap_or(0.), a[1].as_f64().unwrap_or(0.)]).unwrap_or([0., 0.]);
    let anchor = *ids
        .iter()
        .min_by(|&&p, &&q| {
            let dp = (mesh.nodes[&p][0] - anc[0]).powi(2) + (mesh.nodes[&p][1] - anc[1]).powi(2);
            let dq = (mesh.nodes[&q][0] - anc[0]).powi(2) + (mesh.nodes[&q][1] - anc[1]).powi(2);
            dp.partial_cmp(&dq).unwrap()
        })
        .ok_or("empty mesh")?;
    let _ = writeln!(d, "FINISH\n/GOPR\n/SOLU\nANTYPE,STATIC\nD,{anchor},UX,0\nD,{anchor},UY,0");
    for (k, c) in cases.iter().enumerate() {
        let (exx, eyy, exy) = (c[0], c[1], c[2] / 2.);
        let _ = writeln!(d, "/COM, strain case {}: exx={exx} eyy={eyy} gxy={}\nCEDELE,ALL", k + 1, c[2]);
        for &(s, m) in &pr {
            let _ = writeln!(d, "CE,NEXT,{:.15e},{s},UX,1,{m},UX,-1\nCE,NEXT,{:.15e},{s},UY,1,{m},UY,-1", exx * l, exy * l);
        }
        for &(s, m) in &pt {
            let _ = writeln!(d, "CE,NEXT,{:.15e},{s},UX,1,{m},UX,-1\nCE,NEXT,{:.15e},{s},UY,1,{m},UY,-1", exy * l, eyy * l);
        }
        let _ = writeln!(d, "SOLVE");
    }
    let _ = writeln!(d, "FINISH");
    std::fs::write(dir.join("job_local.dat"), &d).map_err(|e| e.to_string())?;
    Ok(Deck { text: d, cases, l, h, npr: pr.len(), npt: pt.len() })
}

/// run one bundle of the rebuilt solver; writes job_local.dat, job_local.out and results.json
pub fn run(dir: &Path, threads: usize, log: &mut String) -> Result<(), String> {
    let Deck { text: d, cases, l, h, npr, npt } = make_deck(dir, log)?;
    // ---- solve in-process
    let mut s = Session::new(Settings { threads: threads.max(1), solver: SolverKind::Sparse, ..Default::default() });
    s.dir = dir.to_path_buf();
    s.run_text(&d);
    let _ = std::fs::write(dir.join("job_local.out"), &s.out);
    if !s.errors.is_empty() {
        return Err(format!("rebuilt solver: {}", s.errors.join(" | ")));
    }
    if s.ce_forces.len() != cases.len() {
        return Err(format!("rebuilt solver produced {} solution(s) for {} strain case(s)", s.ce_forces.len(), cases.len()));
    }
    // ---- effective stress, Hill's lemma: Sigma_ij = (1/V) sum over constraints (x_s - x_m)_i F_j
    let v = l * l * h;
    let mut averages = Vec::new();
    for forces in &s.ce_forces {
        let mut sg = [[0.; 2]; 2];
        for (ce, fs) in forces {
            let (xs, xm) = (s.model.node(ce.slave).unwrap_or([0.; 3]), s.model.node(ce.master).unwrap_or([0.; 3]));
            for i in 0..2 {
                sg[i][ce.dof as usize] += (xs[i] - xm[i]) * fs / v;
            }
        }
        averages.push([sg[0][0], sg[1][1], 0.5 * (sg[0][1] + sg[1][0])]);
    }
    let st = s.results.last().map(|r| r.stats.clone()).unwrap_or_default();
    let out = serde_json::json!({
        "schema": "fem-client/raw@1",
        "engine": "rebuilt_local",
        "engine_version": format!("mapdl_fast {} (rebuilt MAPDL-compatible solver, in-process)", env!("CARGO_PKG_VERSION")),
        "strain_cases": cases,
        "stress_averages": averages,
        "pairing": "CE node pairs, penalty coupling in the rebuilt solver",
        "stress_route": "Hill's lemma over the constraint forces",
        "n_pairs": [npr, npt],
        "equations": st.equations,
        "solve_seconds": st.t_total,
        "fields": "job_local.out",
    });
    std::fs::write(dir.join("results.json"), serde_json::to_string_pretty(&out).unwrap_or_default()).map_err(|e| e.to_string())?;
    Ok(())
}
