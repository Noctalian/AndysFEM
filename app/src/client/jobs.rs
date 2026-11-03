//! jobs: job bundles (self-contained record of inputs, versions and outputs), parametric sweeps,
//! and a run queue limited by the RAM budget and the MAPDL licence seats.

use super::adapters::{job_files, Engine, EngineSettings};
use super::model::*;
use super::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub from: f64,
    pub to: f64,
    pub steps: usize,
}

impl Range {
    pub fn values(&self) -> Vec<f64> {
        if self.steps <= 1 {
            return vec![self.from];
        }
        (0..self.steps).map(|i| self.from + (self.to - self.from) * i as f64 / (self.steps - 1) as f64).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sweep {
    pub name: String,
    pub base: Job,
    pub theta_deg: Range,
    pub t: Range,
    pub engines: Vec<Engine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Machine {
    pub ram_budget_mb: u64,
    pub ram_per_engine_mb: BTreeMap<Engine, u64>,
    pub mapdl_seats: usize,
    pub cores: usize,
}

impl Default for Machine {
    fn default() -> Self {
        let mut r = BTreeMap::new();
        r.insert(Engine::Local, 800);
        r.insert(Engine::CodeAster, 1800);
        r.insert(Engine::Fenicsx, 1500);
        r.insert(Engine::Mapdl, 2500);
        Machine {
            ram_budget_mb: 6144, // 8 GB laptop minus desktop and client
            ram_per_engine_mb: r,
            mapdl_seats: 1,
            cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2),
        }
    }
}

impl Machine {
    /// how many runs of this engine may run at once
    pub fn max_parallel(&self, e: Engine) -> usize {
        let per = *self.ram_per_engine_mb.get(&e).unwrap_or(&2000);
        let by_ram = (self.ram_budget_mb / per.max(1)) as usize;
        let mut n = by_ram.min(self.cores).max(1);
        if e == Engine::Mapdl {
            n = n.min(self.mapdl_seats.max(1));
        }
        n
    }
}

pub struct Expanded {
    pub jobs: Vec<Job>,
    pub rejected: Vec<(String, Vec<String>)>,
}

impl Sweep {
    pub fn expand(&self) -> Expanded {
        let mut jobs = Vec::new();
        let mut rejected = Vec::new();
        let thetas = if self.base.cell.kind == CellKind::Homogeneous { vec![self.base.cell.theta_deg] } else { self.theta_deg.values() };
        let ts = if self.base.cell.kind == CellKind::Homogeneous { vec![self.base.cell.t] } else { self.t.values() };
        for &th in &thetas {
            for &t in &ts {
                let mut j = self.base.clone();
                j.cell.theta_deg = th;
                j.cell.t = t;
                let f = j.cell.feasibility(&j.mesh);
                if f.is_empty() {
                    jobs.push(j);
                } else {
                    rejected.push((format!("theta {th:.3} deg, t {t:.4}"), f));
                }
            }
        }
        Expanded { jobs, rejected }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Written,
    Queued,
    Running,
    Done,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Bundle {
    pub dir: PathBuf,
    pub case_id: String,
    pub engine: Engine,
    pub job: Job,
    pub status: Status,
}

pub fn bundle_dir(workspace: &Path, sweep: &str, case_id: &str, e: Engine) -> PathBuf {
    workspace.join("bundles").join(sweep).join(case_id).join(e.key())
}

/// write one self-contained job bundle: params, engine files, manifest with hashes and pins
pub fn write_bundle(workspace: &Path, sweep: &str, job: &Job, e: Engine, s: &EngineSettings) -> std::io::Result<Bundle> {
    let case_id = job.case_id();
    let dir = bundle_dir(workspace, sweep, &case_id, e);
    std::fs::create_dir_all(&dir)?;
    let params = serde_json::to_string_pretty(&job.params_json()).unwrap_or_default();
    let mut files: Vec<(String, String)> = vec![("params.json".into(), params)];
    for (n, c) in job_files(e, s) {
        files.push((n.to_string(), c));
    }
    let mut hashes = serde_json::Map::new();
    for (n, c) in &files {
        std::fs::write(dir.join(n), c)?;
        hashes.insert(n.clone(), serde_json::Value::String(sha256_hex(c.as_bytes())));
    }
    let image = match e {
        Engine::Local => format!("in-process (mapdl_fast {})", env!("CARGO_PKG_VERSION")),
        Engine::CodeAster => s.aster_image.clone(),
        Engine::Fenicsx => s.fenicsx_image.clone(),
        Engine::Mapdl => format!("school container at {}:{}", s.mapdl_ip, s.mapdl_port),
    };
    let unpinned: Vec<&str> = [&s.gmsh_pin, &s.meshlane_pin, &s.pymapdl_pin]
        .iter()
        .filter(|p| !p.contains("=="))
        .map(|p| p.as_str())
        .collect();
    let manifest = serde_json::json!({
        "schema": "fem-client/bundle@1",
        "client_version": env!("CARGO_PKG_VERSION"),
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "sweep": sweep,
        "case_id": case_id,
        "engine": e.key(),
        "engine_role": e.role(),
        "engine_image": image,
        "container_runtime": s.runtime,
        "pins": {"gmsh": s.gmsh_pin, "meshlane": s.meshlane_pin, "pymapdl": s.pymapdl_pin},
        "unpinned": unpinned,
        "periodic_pairing": e.pairing(),
        "stress_route": e.stress_route(),
        "mesh": {"size": job.mesh.size, "order": job.mesh.order, "seed": job.mesh.seed, "format": e.mesh_format()},
        "job": job,
        "files_sha256": hashes,
        "outputs": ["geometry.json", "mesh_check.json", "results.json", "log.txt"],
    });
    std::fs::write(dir.join("bundle.json"), serde_json::to_string_pretty(&manifest).unwrap_or_default())?;
    Ok(Bundle { dir, case_id, engine: e, job: job.clone(), status: Status::Written })
}

/// read back all bundles of a workspace
pub fn scan(workspace: &Path) -> Vec<Bundle> {
    let mut out = Vec::new();
    let root = workspace.join("bundles");
    let Ok(sweeps) = std::fs::read_dir(&root) else { return out };
    for sw in sweeps.flatten() {
        let Ok(cases) = std::fs::read_dir(sw.path()) else { continue };
        for c in cases.flatten() {
            let Ok(engines) = std::fs::read_dir(c.path()) else { continue };
            for e in engines.flatten() {
                let man = e.path().join("bundle.json");
                let Ok(txt) = std::fs::read_to_string(&man) else { continue };
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else { continue };
                let Some(job) = v.get("job").and_then(|j| serde_json::from_value::<Job>(j.clone()).ok()) else { continue };
                let engine = Engine::ALL.into_iter().find(|x| Some(x.key()) == v.get("engine").and_then(|s| s.as_str()));
                let Some(engine) = engine else { continue };
                let dir = e.path();
                let status = if dir.join("results.json").exists() {
                    Status::Done
                } else if let Ok(f) = std::fs::read_to_string(dir.join("failed.txt")) {
                    Status::Failed(f)
                } else {
                    Status::Written
                };
                out.push(Bundle { dir, case_id: v["case_id"].as_str().unwrap_or("").to_string(), engine, job, status });
            }
        }
    }
    out.sort_by(|a, b| (a.case_id.clone(), a.engine).cmp(&(b.case_id.clone(), b.engine)));
    out
}

/// run bundles with at most `machine.max_parallel(engine)` concurrent runs per engine.
/// Each run is `bash run.sh` in the bundle folder; output goes to log.txt.
pub fn run_all(bundles: Vec<Bundle>, machine: Machine, progress: Arc<Mutex<Vec<(PathBuf, Status)>>>) {
    use std::collections::VecDeque;
    let queue: Arc<Mutex<VecDeque<Bundle>>> = Arc::new(Mutex::new(bundles.into_iter().collect()));
    let running: Arc<Mutex<BTreeMap<Engine, usize>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let total_slots = Engine::ALL.iter().map(|e| machine.max_parallel(*e)).max().unwrap_or(1);
    let mut handles = Vec::new();
    for _ in 0..total_slots {
        let (q, r, p, m) = (queue.clone(), running.clone(), progress.clone(), machine.clone());
        handles.push(std::thread::spawn(move || loop {
            // take the first queued bundle whose engine still has a free slot
            let next = {
                let mut q = q.lock().unwrap();
                let mut r = r.lock().unwrap();
                let pos = q.iter().position(|b| *r.get(&b.engine).unwrap_or(&0) < m.max_parallel(b.engine));
                match pos {
                    Some(i) => {
                        let b = q.remove(i).unwrap();
                        *r.entry(b.engine).or_insert(0) += 1;
                        Some(b)
                    }
                    None if q.is_empty() => return,
                    None => None,
                }
            };
            let Some(b) = next else {
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            };
            set(&p, &b.dir, Status::Running);
            let _ = std::fs::remove_file(b.dir.join("failed.txt"));
            let status = if b.engine == Engine::Local {
                let mut log = String::new();
                let threads = (m.cores / m.max_parallel(Engine::Local)).max(1);
                let r = super::local::run(&b.dir, threads, &mut log);
                let _ = std::fs::write(b.dir.join("log.txt"), &log);
                match r {
                    Ok(()) => Status::Done,
                    Err(e) => Status::Failed(e),
                }
            } else {
            let out = std::process::Command::new("bash").arg("run.sh").current_dir(&b.dir).output();
            match out {
                Ok(o) => {
                    let mut log = String::from_utf8_lossy(&o.stdout).to_string();
                    log.push_str(&String::from_utf8_lossy(&o.stderr));
                    let _ = std::fs::write(b.dir.join("log.txt"), &log);
                    if o.status.success() && b.dir.join("results.json").exists() {
                        Status::Done
                    } else {
                        let last: String = log.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
                        Status::Failed(format!("exit {:?}: {last}", o.status.code()))
                    }
                }
                Err(e) => Status::Failed(format!("cannot start bash: {e} (runs need Linux with uv and a container runtime)")),
            }
            };
            if let Status::Failed(f) = &status {
                let _ = std::fs::write(b.dir.join("failed.txt"), f);
            }
            set(&p, &b.dir, status);
            *r.lock().unwrap().entry(b.engine).or_insert(1) -= 1;
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn set(p: &Arc<Mutex<Vec<(PathBuf, Status)>>>, d: &Path, s: Status) {
    let mut g = p.lock().unwrap();
    if let Some(x) = g.iter_mut().find(|x| x.0 == d) {
        x.1 = s;
    } else {
        g.push((d.to_path_buf(), s));
    }
}
