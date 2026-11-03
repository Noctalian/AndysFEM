//! Desktop window: input editor, run controls, 3-D result view, listings, speed comparison.

use crate::apdl::Session;
use crate::elements;
use crate::examples;
use crate::model::*;
use crate::post;
use crate::solver::{ResultSet, Settings, SolverKind};
use crate::client::adapters::Engine;
use crate::Cli;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PALETTE: [Color32; 9] = [
    Color32::from_rgb(0, 0, 255),
    Color32::from_rgb(0, 178, 255),
    Color32::from_rgb(0, 255, 255),
    Color32::from_rgb(0, 255, 178),
    Color32::from_rgb(0, 255, 0),
    Color32::from_rgb(178, 255, 0),
    Color32::from_rgb(255, 255, 0),
    Color32::from_rgb(255, 178, 0),
    Color32::from_rgb(255, 0, 0),
];

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    View,
    Listing,
    Table,
    Meshing,
    Study,
    Speed,
    Integrate,
}

#[derive(Clone, Copy, PartialEq)]
enum Field {
    USum,
    UX,
    UY,
    UZ,
    RotSum,
    SEqv,
    S1,
    S3,
    SInt,
    SX,
    SY,
    SZ,
    SXY,
    Reaction,
    Applied,
}

impl Field {
    fn label(self) -> &'static str {
        match self {
            Field::USum => "Total displacement (USUM)",
            Field::UX => "Displacement X (UX)",
            Field::UY => "Displacement Y (UY)",
            Field::UZ => "Displacement Z (UZ)",
            Field::SEqv => "von Mises stress (SEQV)",
            Field::S1 => "1st principal stress (S1)",
            Field::SX => "Stress X (SX)",
            Field::RotSum => "Total rotation (RSUM)",
            Field::S3 => "3rd principal stress (S3)",
            Field::SInt => "Stress intensity (SINT)",
            Field::SY => "Stress Y (SY)",
            Field::SZ => "Stress Z (SZ)",
            Field::SXY => "Shear stress XY (SXY)",
            Field::Reaction => "Reaction forces (arrows)",
            Field::Applied => "Applied forces (arrows)",
        }
    }
    const ALL: [Field; 15] = [
        Field::USum, Field::UX, Field::UY, Field::UZ, Field::RotSum, Field::SEqv, Field::S1, Field::S3, Field::SInt,
        Field::SX, Field::SY, Field::SZ, Field::SXY, Field::Reaction, Field::Applied,
    ];
    fn is_force(self) -> bool {
        matches!(self, Field::Reaction | Field::Applied)
    }
    fn item(self) -> (&'static str, &'static str) {
        match self {
            Field::USum => ("U", "SUM"),
            Field::UX => ("U", "X"),
            Field::UY => ("U", "Y"),
            Field::UZ => ("U", "Z"),
            Field::SEqv => ("S", "EQV"),
            Field::S1 => ("S", "1"),
            Field::SX => ("S", "X"),
            Field::RotSum => ("ROT", "SUM"),
            Field::S3 => ("S", "3"),
            Field::SInt => ("S", "INT"),
            Field::SY => ("S", "Y"),
            Field::SZ => ("S", "Z"),
            Field::SXY => ("S", "XY"),
            Field::Reaction | Field::Applied => ("F", ""),
        }
    }
}

struct Snapshot {
    model: Model,
    result: Option<ResultSet>,
    title: String,
    jobname: String,
    warnings: Vec<String>,
    errors: usize,
    faces: Vec<Vec<usize>>,
    lines: Vec<[usize; 2]>,
    center: [f64; 3],
    radius: f64,
    flat: bool,
    wall: f64,
    unsupported: Vec<String>,
}

struct RunDone {
    session: Session,
    wall: f64,
}

pub struct App {
    text: String,
    path: Option<PathBuf>,
    parallel: bool,
    threads: usize,
    solver: SolverKind,
    running: Option<(Receiver<RunDone>, Arc<AtomicBool>, Instant)>,
    snap: Option<Snapshot>,
    listing: String,
    tab: Tab,
    field: Field,
    deform: f32,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan: Vec2,
    status: String,
    bench_div: usize,
    bench_quad: bool,
    orig_path: String,
    orig_result: Arc<Mutex<Option<String>>>,
    orig_running: Arc<AtomicBool>,
    history: Vec<(String, f64, usize, String)>,
    shot: Option<PathBuf>,
    frames: u64,
    shot_requested: bool,
    shot_done: Arc<AtomicBool>,
    dirty: bool,
    client: crate::client::ui::ClientUi,
    /// solver selected in the toolbar: the rebuilt solver or one of the client engines
    engine: Engine,
    mesh_form: crate::meshing_ui::MeshForm,
}

pub fn run(cli: Cli) -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("fem-client — multi-solver FE client"),
        ..Default::default()
    };
    eframe::run_native("fem-client", opts, Box::new(move |cc| Ok(Box::new(App::new(cc, cli)))))
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, cli: Cli) -> Self {
        let mut v = egui::Visuals::light();
        v.panel_fill = Color32::from_rgb(228, 231, 236);
        v.window_fill = Color32::from_rgb(240, 242, 245);
        v.extreme_bg_color = Color32::from_rgb(252, 252, 253);
        v.selection.bg_fill = Color32::from_rgb(40, 96, 170);
        v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
        cc.egui_ctx.set_visuals(v);
        let mut text = examples::SOLID_BLOCK.to_string();
        let mut path = None;
        if let Some(p) = &cli.input {
            if let Ok(t) = std::fs::read_to_string(p) {
                text = t;
                path = Some(p.clone());
            }
        }
        let mut app = App {
            text,
            path,
            parallel: true,
            threads: cli.threads.unwrap_or_else(crate::default_threads),
            solver: cli.solver.unwrap_or(SolverKind::Sparse),
            running: None,
            snap: None,
            listing: String::new(),
            tab: Tab::View,
            field: Field::USum,
            deform: 1.0,
            yaw: -0.6,
            pitch: 0.45,
            zoom: 1.0,
            pan: Vec2::ZERO,
            status: "Ready.".into(),
            bench_div: 40,
            bench_quad: true,
            orig_path: r"C:\Program Files\ANSYS Inc\v221\ansys\bin\winx64\ANSYS.exe".into(),
            orig_result: Arc::new(Mutex::new(None)),
            orig_running: Arc::new(AtomicBool::new(false)),
            history: Vec::new(),
            shot: cli.screenshot,
            frames: 0,
            shot_requested: false,
            shot_done: Arc::new(AtomicBool::new(false)),
            dirty: false,
            client: crate::client::ui::ClientUi::new(),
            engine: Engine::Local,
            mesh_form: Default::default(),
        };
        app.start_run(&cc.egui_ctx);
        // FEM_TAB=meshing|study|listing opens the window on that tab (used for checks and scripts)
        match std::env::var("FEM_TAB").unwrap_or_default().as_str() {
            "meshing" => app.tab = Tab::Meshing,
            "study" => app.tab = Tab::Study,
            "listing" => app.tab = Tab::Listing,
            _ => {}
        }
        app
    }

    fn settings(&self) -> Settings {
        Settings { threads: if self.parallel { self.threads.max(1) } else { 1 }, solver: self.solver, ..Default::default() }
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        if self.engine != Engine::Local {
            // the code is interpreted into the model, exported to the selected engine and run there
            let (text, engine) = (self.text.clone(), self.engine);
            let (settings, ws) = (self.client.engine_settings(), self.client.workspace());
            let (tx, rx) = channel();
            let ctx2 = ctx.clone();
            let cancel = Arc::new(AtomicBool::new(false));
            std::thread::spawn(move || {
                let t = Instant::now();
                let mut s = crate::client::export::run_on_engine(&text, engine, &settings, &ws);
                let wall = t.elapsed().as_secs_f64();
                s.out.push_str(&format!("\n Run on {} finished in {:.3} s wall clock\n", engine.label(), wall));
                let _ = tx.send(RunDone { session: s, wall });
                ctx2.request_repaint();
            });
            self.status = format!("Running the code on {}…", self.engine.label());
            self.running = Some((rx, cancel, Instant::now()));
            return;
        }
        let text = self.text.clone();
        let settings = self.settings();
        let dir = self.path.as_ref().and_then(|p| p.parent().map(|x| x.to_path_buf()));
        let jobname = self.path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "file".into());
        let (tx, rx) = channel();
        let mut s = Session::new(settings.clone());
        if let Some(d) = dir {
            s.dir = d;
        }
        s.jobname = jobname;
        let cancel = s.cancel.clone();
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            s.out.push_str(&format!(" MAPDL-compatible fast solver\n Parallel threads: {}\n\n", settings.threads));
            s.run_text(&text);
            let wall = t.elapsed().as_secs_f64();
            s.out.push_str(&format!("\n Run finished: {} commands, {} warnings, {} errors, {:.3} s wall clock\n", s.commands_run, s.warnings.len(), s.errors.len(), wall));
            if !s.unsupported.is_empty() {
                let l: Vec<String> = s.unsupported.iter().map(|(k, v)| format!("{k} ({v}x)")).collect();
                s.out.push_str(&format!(" Skipped commands: {}\n", l.join(", ")));
            }
            let _ = tx.send(RunDone { session: s, wall });
            ctx2.request_repaint();
        });
        self.status = format!("Solving with {} thread(s)…", settings.threads);
        self.running = Some((rx, cancel, Instant::now()));
    }

    fn poll_run(&mut self) {
        let done = match &self.running {
            Some((rx, _, _)) => rx.try_recv().ok(),
            None => None,
        };
        if let Some(d) = done {
            self.running = None;
            let s = d.session;
            self.listing = s.out.clone();
            let res = s.result().cloned();
            let errors = s.errors.len();
            self.status = match (&res, errors) {
                (Some(r), 0) => format!(
                    "Solved {} equations in {:.3} s ({} thread(s), {}). Whole run {:.3} s.",
                    r.stats.equations, r.stats.t_total, r.stats.threads, r.stats.solver, d.wall
                ),
                (Some(_), e) => format!("Solved, but {e} error(s) in the input — see Output listing."),
                (None, 0) => format!("Run finished in {:.3} s (no SOLVE in this input).", d.wall),
                (None, e) => format!("{e} error(s) — see Output listing."),
            };
            if let Some(r) = &res {
                let name = self.path.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| s.title.clone());
                self.history.push((name, r.stats.t_total, r.stats.threads, r.stats.solver.clone()));
            }
            let mut unsupported: Vec<String> = s.unsupported.iter().map(|(k, v)| format!("{k} ×{v}")).collect();
            unsupported.truncate(30);
            let (faces, lines) = surface(&s.model);
            let (lo, hi) = s.model.bbox();
            let center = [(lo[0] + hi[0]) / 2., (lo[1] + hi[1]) / 2., (lo[2] + hi[2]) / 2.];
            let radius = (((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() / 2.).max(1e-9);
            let flat = (hi[2] - lo[2]).abs() < 1e-12 * radius.max(1.);
            if flat && self.snap.as_ref().map(|x| !x.flat).unwrap_or(true) {
                self.yaw = 0.;
                self.pitch = 0.;
            } else if !flat && self.snap.as_ref().map(|x| x.flat).unwrap_or(false) {
                self.yaw = -0.6;
                self.pitch = 0.45;
            }
            if let Some(r) = &res {
                let mx = s.model.node_ids().map(|n| r.usum(n)).fold(0., f64::max);
                if mx > 0. {
                    let sc = 0.08 * radius / mx;
                    let mag = 10f64.powf(sc.log10().floor());
                    self.deform = ((sc / mag).round().max(1.) * mag) as f32;
                }
            }
            if res.is_none() || errors > 0 {
                if res.is_none() {
                    self.tab = Tab::Listing;
                }
            }
            self.snap = Some(Snapshot {
                title: s.title.clone(),
                jobname: s.jobname.clone(),
                warnings: s.warnings.clone(),
                errors,
                model: s.model,
                result: res,
                faces,
                lines,
                center,
                radius,
                flat,
                wall: d.wall,
                unsupported,
            });
        }
    }

    fn open_file(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("MAPDL input", &["dat", "inp", "mac", "txt", "ans", "cdb"])
            .add_filter("All files", &["*"])
            .pick_file()
        {
            match std::fs::read_to_string(&p) {
                Ok(t) => {
                    self.text = t;
                    self.status = format!("Opened {}", p.display());
                    self.path = Some(p);
                    self.dirty = false;
                }
                Err(e) => self.status = format!("Cannot open: {e}"),
            }
        }
    }

    fn save_file(&mut self, ask: bool) {
        let p = if ask || self.path.is_none() {
            rfd::FileDialog::new().add_filter("MAPDL input", &["dat", "inp", "mac"]).set_file_name("model.dat").save_file()
        } else {
            self.path.clone()
        };
        if let Some(p) = p {
            match std::fs::write(&p, &self.text) {
                Ok(_) => {
                    self.status = format!("Saved {}", p.display());
                    self.path = Some(p);
                    self.dirty = false;
                }
                Err(e) => self.status = format!("Cannot save: {e}"),
            }
        }
    }

    fn export(&mut self, kind: &str) {
        let Some(s) = &self.snap else { return };
        let (ext, content) = match (kind, &s.result) {
            ("json", Some(r)) => ("json", post::to_json(&s.model, r, &s.jobname, &s.title, &s.warnings)),
            ("csv", Some(r)) => ("csv", post::to_csv(&s.model, r)),
            ("out", _) => ("out", self.listing.clone()),
            _ => {
                self.status = "No results to export yet.".into();
                return;
            }
        };
        if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{}.{ext}", s.jobname)).save_file() {
            self.status = match std::fs::write(&p, content) {
                Ok(_) => format!("Exported {}", p.display()),
                Err(e) => format!("Cannot write: {e}"),
            };
        }
    }

    fn time_original(&mut self) {
        if self.orig_running.load(Ordering::Relaxed) {
            return;
        }
        let exe = PathBuf::from(self.orig_path.trim());
        if !exe.exists() {
            *self.orig_result.lock().unwrap() = Some(format!("Not found: {}", exe.display()));
            return;
        }
        let dir = std::env::temp_dir().join("mapdl_fast_compare");
        let _ = std::fs::create_dir_all(&dir);
        let inp = dir.join("compare.dat");
        if std::fs::write(&inp, &self.text).is_err() {
            *self.orig_result.lock().unwrap() = Some("Cannot write the temporary input file.".into());
            return;
        }
        let np = if self.parallel { self.threads.max(1) } else { 1 };
        let flag = self.orig_running.clone();
        let out = self.orig_result.clone();
        flag.store(true, Ordering::Relaxed);
        *out.lock().unwrap() = Some(format!("Running the original with -np {np}… (needs a licence; this can take a while)"));
        std::thread::spawn(move || {
            let t = Instant::now();
            let r = std::process::Command::new(&exe)
                .args(["-b", "-i"])
                .arg(&inp)
                .arg("-o")
                .arg(dir.join("compare.out"))
                .args(["-np", &np.to_string(), "-dir"])
                .arg(&dir)
                .args(["-j", "compare"])
                .status();
            let w = t.elapsed().as_secs_f64();
            *out.lock().unwrap() = Some(match r {
                Ok(st) => format!(
                    "Original finished in {w:.2} s (exit code {}). Its listing: {}",
                    st.code().unwrap_or(-1),
                    dir.join("compare.out").display()
                ),
                Err(e) => format!("Could not start the original: {e}"),
            });
            flag.store(false, Ordering::Relaxed);
        });
    }
}

/// exterior faces (corner nodes) and line elements for drawing
fn surface(m: &Model) -> (Vec<Vec<usize>>, Vec<[usize; 2]>) {
    let mut count: HashMap<Vec<usize>, (u32, Vec<usize>)> = HashMap::new();
    let mut flat = Vec::new();
    let mut lines = Vec::new();
    for id in m.elem_ids() {
        let e = m.elems[id].as_ref().unwrap();
        let Some(k) = m.kind_of(e) else { continue };
        match k {
            Kind::Link | Kind::Beam | Kind::Spring => {
                if e.nodes.len() >= 2 {
                    lines.push([e.nodes[0], e.nodes[1]]);
                }
            }
            Kind::Plane182 | Kind::Plane183 => {
                let mut c: Vec<usize> = e.nodes.iter().take(if k == Kind::Plane183 && e.nodes.len() == 6 { 3 } else { 4 }).copied().collect();
                c.dedup();
                if c.len() >= 3 {
                    flat.push(c);
                }
            }
            Kind::Solid185 | Kind::Solid186 | Kind::Solid187 => {
                let nc = if k == Kind::Solid187 { 3 } else { 4 };
                for f in elements::faces(m, e) {
                    let mut c: Vec<usize> = f.iter().take(nc).map(|&i| e.nodes[i]).collect();
                    c.dedup();
                    if c.len() > 1 && c[0] == *c.last().unwrap() {
                        c.pop();
                    }
                    if c.len() < 3 {
                        continue;
                    }
                    let mut key = c.clone();
                    key.sort_unstable();
                    let ent = count.entry(key).or_insert((0, c));
                    ent.0 += 1;
                }
            }
            _ => {}
        }
    }
    let mut faces: Vec<Vec<usize>> = count.into_values().filter(|v| v.0 == 1).map(|v| v.1).collect();
    faces.extend(flat);
    (faces, lines)
}

fn band(v: f64, lo: f64, hi: f64) -> usize {
    if hi <= lo {
        return 0;
    }
    (((v - lo) / (hi - lo) * 9.0).floor() as i64).clamp(0, 8) as usize
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_run();
        self.frames += 1;
        if self.running.is_some() || self.orig_running.load(Ordering::Relaxed) || self.shot.is_some() {
            ctx.request_repaint();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.start_run(&ctx);
        }
        // the unit-cell study can hand its APDL deck to the code window
        if let Some(deck) = self.client.request_deck.take() {
            if !deck.is_empty() {
                self.text = deck;
                self.path = None;
            }
            self.tab = Tab::Meshing;
        }
        self.legacy_ui(ui, &ctx);
        self.screenshot(&ctx);
    }
}

impl App {
    fn legacy_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let ctx = ctx.clone();
        // ------------------------------------------------ top bar
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("fem-client").strong().color(Color32::from_rgb(20, 50, 100)));
                ui.separator();
                if ui.button("Open…").clicked() {
                    self.open_file();
                }
                if ui.button("Save").clicked() {
                    self.save_file(false);
                }
                if ui.button("Save as…").clicked() {
                    self.save_file(true);
                }
                ui.menu_button("Examples", |ui| {
                    for (name, txt) in examples::EXAMPLES {
                        if ui.button(name).clicked() {
                            self.text = txt.to_string();
                            self.path = None;
                            ui.close();
                        }
                    }
                });
                ui.separator();
                ui.label("Solver");
                egui::ComboBox::from_id_salt("engine").width(230.0).selected_text(self.engine.label()).show_ui(ui, |ui| {
                    ui.label(egui::RichText::new("Reverse-engineered").small().color(Color32::DARK_GRAY));
                    ui.selectable_value(&mut self.engine, Engine::Local, Engine::Local.label());
                    ui.label(egui::RichText::new("Client engines").small().color(Color32::DARK_GRAY));
                    for e in [Engine::CodeAster, Engine::Fenicsx, Engine::Mapdl] {
                        ui.selectable_value(&mut self.engine, e, e.label()).on_hover_text(e.role());
                    }
                });
                let running = self.running.is_some();
                let run = ui.add_enabled(
                    !running,
                    egui::Button::new(egui::RichText::new("Run  (F5)").strong().color(Color32::WHITE)).fill(Color32::from_rgb(30, 120, 60)),
                );
                if run.clicked() {
                    self.start_run(&ctx);
                }
                if running {
                    if ui.button("Stop").clicked() {
                        if let Some((_, c, _)) = &self.running {
                            c.store(true, Ordering::Relaxed);
                        }
                    }
                    ui.spinner();
                }
                ui.separator();
                let local = self.engine == Engine::Local;
                ui.add_enabled(local, egui::Checkbox::new(&mut self.parallel, "Parallel"));
                ui.add_enabled(self.parallel && local, egui::DragValue::new(&mut self.threads).range(1..=256).suffix(" cores"));
                ui.separator();
                ui.add_enabled_ui(local, |ui| ui.label("Equation solver"));
                if local { egui::ComboBox::from_id_salt("solver")
                    .selected_text(if self.solver == SolverKind::Sparse { "Sparse direct" } else { "PCG iterative" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.solver, SolverKind::Sparse, "Sparse direct");
                        ui.selectable_value(&mut self.solver, SolverKind::Pcg, "PCG iterative");
                    }); }
                ui.separator();
                ui.menu_button("Export", |ui| {
                    if ui.button("Results as JSON…").clicked() {
                        self.export("json");
                        ui.close();
                    }
                    if ui.button("Nodal results as CSV…").clicked() {
                        self.export("csv");
                        ui.close();
                    }
                    if ui.button("Output listing (.out)…").clicked() {
                        self.export("out");
                        ui.close();
                    }
                });
            });
            ui.add_space(4.0);
        });
        // ------------------------------------------------ status bar
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some((_, _, t)) = &self.running {
                    ui.label(format!("{:.1} s", t.elapsed().as_secs_f64()));
                }
                ui.label(&self.status);
                if let Some(s) = &self.snap {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!(
                            "{} nodes · {} elements · {} warnings · {} errors",
                            s.model.node_count(),
                            s.model.elem_count(),
                            s.warnings.len(),
                            s.errors
                        ));
                    });
                }
            });
        });
        // ------------------------------------------------ editor
        egui::Panel::left("editor").default_size(470.0).resizable(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Code (MAPDL commands)");
                let name = self.path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(not saved)".into());
                ui.label(egui::RichText::new(name).small().color(Color32::DARK_GRAY));
            });
            ui.separator();
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                let r = ui.add(
                    egui::TextEdit::multiline(&mut self.text)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(40)
                        .font(egui::TextStyle::Monospace),
                );
                if r.changed() {
                    self.dirty = true;
                }
            });
        });
        // ------------------------------------------------ main area
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::View, "Result view");
                ui.selectable_value(&mut self.tab, Tab::Listing, "Output listing");
                ui.selectable_value(&mut self.tab, Tab::Table, "Nodal results");
                ui.selectable_value(&mut self.tab, Tab::Meshing, "Meshing");
                ui.selectable_value(&mut self.tab, Tab::Study, "Unit-cell study");
                ui.selectable_value(&mut self.tab, Tab::Speed, "Speed");
                ui.selectable_value(&mut self.tab, Tab::Integrate, "Use in my processes");
            });
            ui.separator();
            match self.tab {
                Tab::View => self.view_tab(ui),
                Tab::Listing => {
                    egui::ScrollArea::both().auto_shrink([false, false]).stick_to_bottom(false).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(&self.listing).monospace().size(12.0)).extend());
                    });
                }
                Tab::Table => self.table_tab(ui),
                Tab::Meshing => {
                    let ws = self.client.workspace();
                    if self.mesh_form.ui(ui, &mut self.text, ws) {
                        self.dirty = true;
                        self.status = self.mesh_form.last.clone();
                    }
                }
                Tab::Study => self.client.show(ui),
                Tab::Speed => self.speed_tab(ui, &ctx),
                Tab::Integrate => integrate_tab(ui),
            }
        });
    }

    fn screenshot(&mut self, ctx: &egui::Context) {
        let ctx = ctx.clone();
        if let Some(p) = self.shot.clone() {
            if self.shot_done.load(Ordering::Relaxed) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else if !self.shot_requested && self.frames >= 120 && (self.running.is_none() || self.frames > 1200) {
                self.shot_requested = true;
                let done = self.shot_done.clone();
                let c2 = ctx.clone();
                ctx.request_screenshot(move |img: Arc<egui::ColorImage>| {
                    let [w, h] = img.size;
                    let mut buf = Vec::with_capacity(w * h * 4);
                    for px in &img.pixels {
                        buf.extend_from_slice(&[px.r(), px.g(), px.b(), 255]);
                    }
                    if let Some(parent) = p.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if let Some(im) = image::RgbaImage::from_raw(w as u32, h as u32, buf) {
                        let _ = im.save(&p);
                    }
                    done.store(true, Ordering::Relaxed);
                    c2.request_repaint();
                });
            }
        }
    }
}

impl App {
    fn view_tab(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.snap else {
            ui.label("Solve a model to see results.");
            return;
        };
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("field").selected_text(self.field.label()).width(230.0).show_ui(ui, |ui| {
                for f in Field::ALL {
                    ui.selectable_value(&mut self.field, f, f.label());
                }
            });
            ui.label("Deformation ×");
            ui.add(egui::DragValue::new(&mut self.deform).range(0.0..=1.0e9).speed(0.5));
            if ui.button("Auto").clicked() {
                if let Some(r) = &s.result {
                    let mx = s.model.node_ids().map(|n| r.usum(n)).fold(0., f64::max);
                    if mx > 0. {
                        self.deform = (0.08 * s.radius / mx) as f32;
                    }
                }
            }
            ui.separator();
            if ui.button("Iso").clicked() {
                self.yaw = -0.6;
                self.pitch = 0.45;
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
            }
            if ui.button("Front").clicked() {
                self.yaw = 0.;
                self.pitch = 0.;
            }
            if ui.button("Top").clicked() {
                self.yaw = 0.;
                self.pitch = std::f32::consts::FRAC_PI_2;
            }
            if ui.button("Right").clicked() {
                self.yaw = -std::f32::consts::FRAC_PI_2;
                self.pitch = 0.;
            }
            ui.label(egui::RichText::new("drag = rotate, right-drag = pan, wheel = zoom").small().color(Color32::DARK_GRAY));
        });
        let avail = ui.available_size();
        let (resp, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
        let rect = resp.rect;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.yaw += d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.57, 1.57);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            self.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let sc = ui.input(|i| i.smooth_scroll_delta.y);
            if sc != 0. {
                self.zoom = (self.zoom * (1.0 + sc * 0.002)).clamp(0.05, 50.0);
            }
        }
        // background gradient
        let mut mesh = egui::Mesh::default();
        let top = Color32::from_rgb(38, 58, 104);
        let bot = Color32::from_rgb(6, 8, 20);
        mesh.colored_vertex(rect.left_top(), top);
        mesh.colored_vertex(rect.right_top(), top);
        mesh.colored_vertex(rect.right_bottom(), bot);
        mesh.colored_vertex(rect.left_bottom(), bot);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(egui::Shape::mesh(mesh));
        let painter = painter.with_clip_rect(rect);
        // values
        let m = &s.model;
        let (item, comp) = self.field.item();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut val = vec![f64::NAN; m.nodes.len()];
        let mut fvec: Vec<(usize, [f64; 3])> = Vec::new();
        if self.field.is_force() {
            let mut acc: std::collections::BTreeMap<usize, [f64; 3]> = std::collections::BTreeMap::new();
            match self.field {
                Field::Reaction => {
                    if let Some(r) = &s.result {
                        for (&(n, d), &v) in &r.reac {
                            if d < 3 {
                                acc.entry(n).or_insert([0.; 3])[d as usize] += v;
                            }
                        }
                    }
                }
                _ => {
                    for (&(n, d), &v) in &m.force {
                        if d < 3 {
                            acc.entry(n).or_insert([0.; 3])[d as usize] += v;
                        }
                    }
                }
            }
            for (n, f) in acc {
                let mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                if mag > 0. {
                    lo = lo.min(mag);
                    hi = hi.max(mag);
                    fvec.push((n, f));
                }
            }
            if fvec.len() == 1 {
                lo = 0.;
            }
        } else if let Some(r) = &s.result {
            for n in m.node_ids() {
                if let Some(v) = post::node_value(m, r, n, item, comp) {
                    if n < r.mask.len() && (r.mask[n] != 0 || item == "S") {
                        val[n] = v;
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
        }
        let has_vals = lo.is_finite();
        let face_vals = has_vals && !self.field.is_force();
        // projection
        let (cy, sy) = (self.yaw.cos() as f64, self.yaw.sin() as f64);
        let (cp, sp) = (self.pitch.cos() as f64, self.pitch.sin() as f64);
        let scale = (rect.width().min(rect.height()) as f64 * 0.42 / s.radius) * self.zoom as f64;
        let ctr = rect.center() + self.pan + Vec2::new(-40.0, 0.0);
        let defo = self.deform as f64;
        let proj = |n: usize| -> Option<(Pos2, f64)> {
            let x = m.node(n)?;
            let mut p = [x[0] - s.center[0], x[1] - s.center[1], x[2] - s.center[2]];
            if let Some(r) = &s.result {
                for k in 0..3 {
                    p[k] += defo * r.u(n, k as u8);
                }
            }
            // yaw about Y, then pitch about X
            let x1 = cy * p[0] - sy * p[2];
            let z1 = sy * p[0] + cy * p[2];
            let y2 = cp * p[1] - sp * z1;
            let z2 = sp * p[1] + cp * z1;
            Some((Pos2::new(ctr.x + (x1 * scale) as f32, ctr.y - (y2 * scale) as f32), z2))
        };
        let mut pts: Vec<Option<(Pos2, f64)>> = vec![None; m.nodes.len()];
        for n in m.node_ids() {
            pts[n] = proj(n);
        }
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(s.faces.len());
        for (i, f) in s.faces.iter().enumerate() {
            let mut z = 0.;
            let mut ok = true;
            for &n in f {
                match pts.get(n).copied().flatten() {
                    Some(p) => z += p.1,
                    None => ok = false,
                }
            }
            if ok {
                order.push((z / f.len() as f64, i));
            }
        }
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let edge = if s.faces.len() > 40_000 { Stroke::NONE } else { Stroke::new(0.6, Color32::from_rgba_unmultiplied(0, 0, 0, 120)) };
        for (_, i) in order.iter().take(400_000) {
            let f = &s.faces[*i];
            let poly: Vec<Pos2> = f.iter().map(|&n| pts[n].unwrap().0).collect();
            let col = if face_vals {
                let vs: Vec<f64> = f.iter().map(|&n| val[n]).filter(|v| v.is_finite()).collect();
                if vs.is_empty() {
                    Color32::from_gray(150)
                } else {
                    PALETTE[band(vs.iter().sum::<f64>() / vs.len() as f64, lo, hi)]
                }
            } else {
                Color32::from_rgb(140, 170, 210)
            };
            painter.add(egui::Shape::convex_polygon(poly, col, edge));
        }
        for l in &s.lines {
            if let (Some(a), Some(b)) = (pts.get(l[0]).copied().flatten(), pts.get(l[1]).copied().flatten()) {
                let col = if face_vals {
                    let v = (val[l[0]] + val[l[1]]) / 2.;
                    if v.is_finite() {
                        PALETTE[band(v, lo, hi)]
                    } else {
                        Color32::from_gray(200)
                    }
                } else {
                    Color32::from_gray(220)
                };
                painter.line_segment([a.0, b.0], Stroke::new(3.0, col));
            }
        }
        // constraint markers
        if m.disp.len() < 20_000 {
            for &(n, _) in m.disp.keys() {
                if let Some(Some(p)) = pts.get(n) {
                    painter.circle_filled(p.0, 2.0, Color32::from_rgb(80, 220, 255));
                }
            }
        }
        if m.force.len() < 20_000 {
            for &(n, _) in m.force.keys() {
                if let Some(Some(p)) = pts.get(n) {
                    painter.circle_filled(p.0, 2.5, Color32::from_rgb(255, 80, 200));
                }
            }
        }
        // force arrows (reaction or applied), coloured by magnitude
        for (n, f) in &fvec {
            if let Some(Some(p)) = pts.get(*n) {
                let mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                let x1 = cy * f[0] - sy * f[2];
                let z1 = sy * f[0] + cy * f[2];
                let y2 = cp * f[1] - sp * z1;
                let len = (x1 * x1 + y2 * y2).sqrt().max(1e-300);
                let px = 12.0 + 38.0 * (mag / hi.max(1e-300)) as f32;
                let dir = Vec2::new((x1 / len) as f32, (-y2 / len) as f32) * px;
                let col = PALETTE[band(mag, lo, hi)];
                let tip = p.0 + dir;
                painter.line_segment([p.0, tip], Stroke::new(2.0, col));
                let side = Vec2::new(-dir.y, dir.x) * 0.25;
                painter.line_segment([tip, tip - dir * 0.3 + side], Stroke::new(2.0, col));
                painter.line_segment([tip, tip - dir * 0.3 - side], Stroke::new(2.0, col));
            }
        }
        if self.field.is_force() && fvec.is_empty() {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, if self.field == Field::Reaction { "No reaction forces (solve first)" } else { "No applied forces in the model" }, egui::FontId::proportional(16.0), Color32::from_gray(220));
        }
        // title & legend
        let font = egui::FontId::proportional(14.0);
        let small = egui::FontId::monospace(12.0);
        painter.text(rect.left_top() + Vec2::new(14.0, 12.0), egui::Align2::LEFT_TOP, if s.title.is_empty() { "Untitled model" } else { &s.title }, font.clone(), Color32::WHITE);
        let info = match &s.result {
            Some(r) => format!(
                "{}   ·   {} equations solved in {:.3} s   ·   {} thread(s), {}   ·   deformation ×{}",
                self.field.label(),
                r.stats.equations,
                r.stats.t_total,
                r.stats.threads,
                r.stats.solver,
                self.deform
            ),
            None => "No solution yet (model only)".into(),
        };
        painter.text(rect.left_top() + Vec2::new(14.0, 32.0), egui::Align2::LEFT_TOP, info, egui::FontId::proportional(12.0), Color32::from_gray(210));
        if has_vals {
            let x0 = rect.right() - 150.0;
            let y0 = rect.top() + 70.0;
            let h = 26.0;
            for b in 0..9 {
                let k = 8 - b;
                let r = Rect::from_min_size(Pos2::new(x0, y0 + b as f32 * h), Vec2::new(26.0, h));
                painter.rect_filled(r, 0.0, PALETTE[k]);
                let v = lo + (hi - lo) * (k as f64 + 1.0) / 9.0;
                painter.text(Pos2::new(x0 + 34.0, y0 + b as f32 * h), egui::Align2::LEFT_CENTER, post::g(v).trim(), small.clone(), Color32::WHITE);
            }
            painter.text(Pos2::new(x0 + 34.0, y0 + 9.0 * h), egui::Align2::LEFT_CENTER, post::g(lo).trim(), small.clone(), Color32::WHITE);
            painter.text(Pos2::new(x0, y0 - 18.0), egui::Align2::LEFT_CENTER, format!("MAX {}", post::g(hi).trim()), small.clone(), Color32::from_rgb(255, 140, 140));
            painter.text(Pos2::new(x0, y0 + 9.0 * h + 22.0), egui::Align2::LEFT_CENTER, format!("MIN {}", post::g(lo).trim()), small, Color32::from_rgb(140, 170, 255));
        }
        if !s.unsupported.is_empty() || !s.warnings.is_empty() {
            let mut y = rect.bottom() - 16.0;
            let mut lines: Vec<String> = s.warnings.iter().rev().take(3).map(|w| format!("Warning: {w}")).collect();
            if !s.unsupported.is_empty() {
                lines.insert(0, format!("Skipped commands: {}", s.unsupported.join(", ")));
            }
            for l in lines {
                let l: String = l.chars().take(160).collect();
                painter.text(Pos2::new(rect.left() + 14.0, y), egui::Align2::LEFT_BOTTOM, l, egui::FontId::proportional(12.0), Color32::from_rgb(255, 220, 120));
                y -= 18.0;
            }
        }
        let _ = s.wall;
    }

    fn table_tab(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.snap else {
            ui.label("No results yet.");
            return;
        };
        let Some(r) = &s.result else {
            ui.label("No results yet (the input has no SOLVE).");
            return;
        };
        let ids: Vec<usize> = s.model.node_ids().filter(|&n| n < r.mask.len() && r.mask[n] != 0).collect();
        let head = format!("{:>9} {:>13} {:>13} {:>13} {:>13} {:>13}", "NODE", "UX", "UY", "UZ", "USUM", "SEQV");
        ui.label(egui::RichText::new(head).monospace().strong());
        let row_h = 16.0;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, row_h, ids.len(), |ui, range| {
            for i in range {
                let n = ids[i];
                let sq = r.stress[n].map(|x| elements::von_mises(&x));
                let l = format!(
                    "{:>9} {} {} {} {} {}",
                    n,
                    post::g(r.u(n, 0)),
                    post::g(r.u(n, 1)),
                    post::g(r.u(n, 2)),
                    post::g(r.usum(n)),
                    sq.map(post::g).unwrap_or_else(|| format!("{:>12}", "-"))
                );
                ui.label(egui::RichText::new(l).monospace());
            }
        });
    }

    fn speed_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("Where the time went");
            if let Some(r) = self.snap.as_ref().and_then(|s| s.result.as_ref()) {
                let st = &r.stats;
                let parts = [
                    ("Numbering & sparsity", st.t_setup, Color32::from_rgb(120, 140, 200)),
                    ("Element matrices + assembly", st.t_assembly, Color32::from_rgb(70, 160, 110)),
                    ("Equation solution", st.t_solve, Color32::from_rgb(220, 120, 50)),
                    ("Reactions + stresses", st.t_post, Color32::from_rgb(170, 90, 170)),
                ];
                let total = parts.iter().map(|p| p.1).sum::<f64>().max(1e-9);
                let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(900.0), 30.0), Sense::hover());
                let mut x = rect.left();
                for (_, t, c) in parts {
                    let w = (t / total) as f32 * rect.width();
                    ui.painter().rect_filled(Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(w, rect.height())), 0.0, c);
                    x += w;
                }
                egui::Grid::new("times").striped(true).show(ui, |ui| {
                    for (n, t, c) in parts {
                        ui.colored_label(c, "[ ]");
                        ui.label(n);
                        ui.label(format!("{t:.4} s"));
                        ui.end_row();
                    }
                    ui.label("");
                    ui.strong("Total solve");
                    ui.strong(format!("{:.4} s", st.t_total));
                    ui.end_row();
                    ui.label("");
                    ui.label("Equations / threads / solver");
                    ui.label(format!("{} / {} / {}", st.equations, st.threads, st.solver));
                    ui.end_row();
                    if st.solver == "PCG" {
                        ui.label("");
                        ui.label("PCG iterations / achieved tolerance");
                        ui.label(format!("{} / {:.2e}", st.pcg_iterations, st.pcg_achieved_tol));
                        ui.end_row();
                    }
                });
            } else {
                ui.label("Solve a model to see the timing.");
            }
            ui.add_space(10.0);
            ui.heading("Runs this session");
            egui::Grid::new("hist").striped(true).show(ui, |ui| {
                ui.strong("Model");
                ui.strong("Solve time");
                ui.strong("Threads");
                ui.strong("Solver");
                ui.end_row();
                for (n, t, th, sv) in self.history.iter().rev().take(15) {
                    ui.label(n);
                    ui.label(format!("{t:.3} s"));
                    ui.label(th.to_string());
                    ui.label(sv);
                    ui.end_row();
                }
            });
            ui.add_space(10.0);
            ui.heading("Speed test model");
            ui.horizontal(|ui| {
                ui.label("Elements along the length");
                ui.add(egui::Slider::new(&mut self.bench_div, 4..=120));
                ui.checkbox(&mut self.bench_quad, "20-node bricks");
                let d2 = (self.bench_div / 2).max(1);
                let nn = if self.bench_quad {
                    (2 * self.bench_div + 1) * (2 * d2 + 1) * (2 * d2 + 1)
                } else {
                    (self.bench_div + 1) * (d2 + 1) * (d2 + 1)
                };
                ui.label(format!("≈ {} equations", nn * 3));
                if ui.button("Load into editor and solve").clicked() {
                    self.text = examples::benchmark(self.bench_div, self.bench_quad);
                    self.path = None;
                    self.start_run(ctx);
                }
            });
            ui.add_space(10.0);
            ui.heading("Compare with the original program");
            ui.label("Runs the original solver in batch mode on the input in the editor, with the same number of cores, and times it. It needs the original's licence.");
            ui.horizontal(|ui| {
                ui.label("Original program:");
                ui.add(egui::TextEdit::singleline(&mut self.orig_path).desired_width(520.0));
            });
            let busy = self.orig_running.load(Ordering::Relaxed);
            if ui.add_enabled(!busy, egui::Button::new("Time the original on this input")).clicked() {
                self.time_original();
            }
            if let Some(t) = self.orig_result.lock().unwrap().clone() {
                ui.label(t);
            }
        });
    }
}

fn integrate_tab(ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.heading("Run it from scripts, schedulers and other programs");
        ui.label("The same program runs without a window. It takes the same switches as the original batch launcher, so existing scripts keep working, and it can also write machine-readable results.");
        ui.add_space(6.0);
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "mapdl_fast.exe".into());
        let cmds = [
            ("Batch run, like the original", format!("\"{exe}\" -b -i model.dat -o model.out -np 8")),
            ("Results for Python / Excel / databases", format!("\"{exe}\" -b -i model.dat -o model.out --json results.json --csv nodes.csv")),
            ("Iterative solver, all cores", format!("\"{exe}\" -b -i model.dat --solver pcg")),
            ("Single core (reproducible timing)", format!("\"{exe}\" -b -i model.dat --serial")),
            ("Read the input from a pipe", format!("type model.dat | \"{exe}\" -b > model.out")),
        ];
        for (t, c) in cmds {
            ui.strong(t);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&c).monospace().background_color(Color32::from_rgb(250, 250, 250)));
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(c.clone());
                }
            });
            ui.add_space(4.0);
        }
        ui.add_space(8.0);
        ui.strong("Exit codes");
        ui.label("0 = solved without errors · 1 = errors in the input · 2 = the solution failed (for example a model that is not held in place).");
        ui.add_space(8.0);
        ui.strong("What the JSON file contains");
        ui.label("Maximum displacement and von Mises stress with their nodes, reaction totals, every node's displacements and averaged stresses, reactions per constrained DOF, link/beam element forces, timing and warnings.");
        ui.add_space(8.0);
        ui.strong("Supported in this version");
        ui.label("Linear static structural analysis. Elements LINK180, BEAM188, COMBIN14, PLANE182, PLANE183, SOLID185, SOLID186, SOLID187, SURF154 (pressure), MASS21 (gravity). Loads D, F, SF/SFE pressure, ACEL. Parameters, *DO, *IF, *GET, *DIM, *VWRITE, *CFWRITE, /INPUT, components, NBLOCK/EBLOCK/CMBLOCK (Workbench input files), BLOCK/RECTNG with VMESH/AMESH. Listings PRNSOL, PRRSOL, PRESOL, FSUM. Other commands are listed as skipped in the output.");
    });
}
