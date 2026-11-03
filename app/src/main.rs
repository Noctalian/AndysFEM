#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! MAPDL-compatible fast structural solver: batch command line + desktop window.

mod apdl;
mod client;
mod elements;
mod examples;
mod expr;
mod gui;
mod meshing;
mod meshing_ui;
mod model;
mod post;
mod solver;

use solver::{Settings, SolverKind};
use std::path::PathBuf;

pub struct Cli {
    pub batch: bool,
    pub input: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub threads: Option<usize>,
    pub dir: Option<PathBuf>,
    pub jobname: Option<String>,
    pub json: Option<PathBuf>,
    pub csv: Option<PathBuf>,
    pub solver: Option<SolverKind>,
    pub screenshot: Option<PathBuf>,
    pub gui: bool,
    pub serial: bool,
}

fn parse_args() -> Result<Cli, String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut c = Cli {
        batch: false,
        input: None,
        output: None,
        threads: None,
        dir: None,
        jobname: None,
        json: None,
        csv: None,
        solver: None,
        screenshot: None,
        gui: false,
        serial: false,
    };
    let mut i = 0;
    let val = |i: &mut usize| -> Result<String, String> {
        *i += 1;
        a.get(*i).cloned().ok_or_else(|| format!("missing value after {}", a[*i - 1]))
    };
    while i < a.len() {
        let s = a[i].to_lowercase();
        match s.as_str() {
            "-b" => {
                c.batch = true;
                // "-b nolist" / "-b list"
                if matches!(a.get(i + 1).map(|x| x.to_lowercase()).as_deref(), Some("nolist") | Some("list")) {
                    i += 1;
                }
            }
            "-i" => c.input = Some(PathBuf::from(val(&mut i)?)),
            "-o" => c.output = Some(PathBuf::from(val(&mut i)?)),
            "-np" | "--threads" => c.threads = Some(val(&mut i)?.parse().map_err(|_| "-np needs a number")?),
            "-dir" => c.dir = Some(PathBuf::from(val(&mut i)?)),
            "-j" => c.jobname = Some(val(&mut i)?),
            "--json" => c.json = Some(PathBuf::from(val(&mut i)?)),
            "--csv" => c.csv = Some(PathBuf::from(val(&mut i)?)),
            "--solver" => {
                c.solver = Some(match val(&mut i)?.to_lowercase().as_str() {
                    "pcg" => SolverKind::Pcg,
                    _ => SolverKind::Sparse,
                })
            }
            "--screenshot" => c.screenshot = Some(PathBuf::from(val(&mut i)?)),
            "--gui" | "-g" => c.gui = true,
            "--serial" => c.serial = true,
            "-smp" | "-dis" | "-mpi" | "-acc" | "-s" | "-m" | "-db" | "-p" | "-l" | "-na" | "-machines" | "-t" | "-d" => {
                // accepted for compatibility with the original launcher; values skipped
                if matches!(s.as_str(), "-s" | "-m" | "-db" | "-p" | "-l" | "-na" | "-machines" | "-mpi" | "-acc" | "-t" | "-d") {
                    i += 1;
                }
            }
            "-h" | "--help" | "/?" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            other => {
                if c.input.is_none() && !other.starts_with('-') {
                    c.input = Some(PathBuf::from(&a[i]));
                } else {
                    eprintln!("note: option {} ignored", a[i]);
                }
            }
        }
        i += 1;
    }
    Ok(c)
}

const HELP: &str = "MAPDL-compatible fast solver

  mapdl_fast -b -i model.dat -o model.out [-np N] [-dir folder] [-j jobname]
             [--json results.json] [--csv nodes.csv] [--solver sparse|pcg] [--serial]
  mapdl_fast                 open the window
  mapdl_fast --screenshot shot.png   open the window, solve the example, save a picture and exit

Exit code: 0 solved / no errors, 1 input errors, 2 solution failed.";

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn attach_console() {
    #[cfg(windows)]
    {
        extern "system" {
            fn AttachConsole(pid: u32) -> i32;
        }
        unsafe {
            AttachConsole(0xFFFF_FFFF);
        }
    }
}

fn run_batch(c: &Cli) -> i32 {
    let threads = if c.serial { 1 } else { c.threads.unwrap_or_else(default_threads).max(1) };
    let mut settings = Settings { threads, ..Default::default() };
    if let Some(s) = c.solver {
        settings.solver = s;
    }
    let mut s = apdl::Session::new(settings);
    if let Some(d) = &c.dir {
        s.dir = d.clone();
    }
    if let Some(j) = &c.jobname {
        s.jobname = j.clone();
    }
    let t = std::time::Instant::now();
    s.out.push_str(&format!(
        " MAPDL-compatible fast solver {}\n Parallel threads: {}\n\n",
        env!("CARGO_PKG_VERSION"),
        threads
    ));
    let res = match &c.input {
        Some(p) => {
            let p = if p.is_absolute() || c.dir.is_none() { p.clone() } else { s.dir.join(p) };
            if let Some(par) = p.parent() {
                if c.dir.is_none() && !par.as_os_str().is_empty() {
                    s.dir = par.to_path_buf();
                }
            }
            s.run_file(&p)
        }
        None => {
            use std::io::Read;
            let mut txt = String::new();
            let _ = std::io::stdin().read_to_string(&mut txt);
            s.run_text(&txt);
            Ok(())
        }
    };
    if let Err(e) = res {
        eprintln!("{e}");
        return 1;
    }
    s.out.push_str(&format!(
        "\n Run finished: {} commands, {} warnings, {} errors, {:.3} s wall clock\n",
        s.commands_run,
        s.warnings.len(),
        s.errors.len(),
        t.elapsed().as_secs_f64()
    ));
    if !s.unsupported.is_empty() {
        let l: Vec<String> = s.unsupported.iter().map(|(k, v)| format!("{k} ({v}x)")).collect();
        s.out.push_str(&format!(" Skipped commands: {}\n", l.join(", ")));
    }
    match &c.output {
        Some(o) => {
            if let Err(e) = std::fs::write(o, &s.out) {
                eprintln!("cannot write {}: {e}", o.display());
            }
        }
        None => print!("{}", s.out),
    }
    if let Some(r) = s.result() {
        if let Some(j) = &c.json {
            let _ = std::fs::write(j, post::to_json(&s.model, r, &s.jobname, &s.title, &s.warnings));
        }
        if let Some(cv) = &c.csv {
            let _ = std::fs::write(cv, post::to_csv(&s.model, r));
        }
    }
    if s.errors.iter().any(|e| e.starts_with("SOLUTION FAILED")) {
        2
    } else if !s.errors.is_empty() {
        1
    } else {
        0
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(|s| s.as_str()) == Some("fem") {
        attach_console();
        std::process::exit(client::cli::main(&raw[1..]));
    }
    let cli = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            attach_console();
            eprintln!("{e}\n\n{HELP}");
            std::process::exit(1);
        }
    };
    if (cli.batch || cli.input.is_some()) && !cli.gui && cli.screenshot.is_none() {
        attach_console();
        std::process::exit(run_batch(&cli));
    }
    if let Err(e) = gui::run(cli) {
        eprintln!("window error: {e}");
        std::process::exit(3);
    }
}
