//! The client window: neutral model on the left, cell preview, job bundles, cross-engine comparison
//! and the verification ladder in the middle.

use super::adapters::{Engine, EngineSettings};
use super::compare::{self, Effective};
use super::jobs::{self, Bundle, Machine, Range, Status, Sweep};
use super::model::*;
use eframe::egui;
use egui::{Color32, Pos2, Sense, Stroke, Vec2};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Cell,
    Jobs,
    Compare,
    Verify,
    Plan,
}

pub struct ClientUi {
    job: Job,
    sweep_name: String,
    theta: Range,
    t: Range,
    use_sweep: bool,
    engines: BTreeMap<Engine, bool>,
    settings: EngineSettings,
    machine: Machine,
    workspace: String,
    tol: f64,
    mixed_exx: f64,
    tab: Tab,
    bundles: Vec<Bundle>,
    progress: Arc<Mutex<Vec<(PathBuf, Status)>>>,
    running: Option<std::thread::JoinHandle<()>>,
    selected_case: Option<String>,
    status: String,
    rejected: Vec<(String, Vec<String>)>,
    /// the solver selected on the neutral model
    pub active: Engine,
    /// set when the person asks to open the rebuilt solver's APDL deck editor (deck text)
    pub request_deck: Option<String>,
}

fn ok_bundles(f: &[String], sweep: bool) -> bool {
    f.is_empty() || sweep
}

impl ClientUi {
    pub fn new() -> Self {
        // extra solvers for the cross-check (the active solver is chosen with the selector)
        let mut engines = BTreeMap::new();
        engines.insert(Engine::Local, false);
        engines.insert(Engine::CodeAster, false);
        engines.insert(Engine::Fenicsx, false);
        engines.insert(Engine::Mapdl, false);
        let workspace = std::env::current_dir().unwrap_or_default().join("fem-workspace").display().to_string();
        let mut s = ClientUi {
            job: super::default_job(),
            sweep_name: "single".into(),
            theta: Range { from: 5.0, to: 25.0, steps: 5 },
            t: Range { from: 0.6, to: 1.4, steps: 3 },
            use_sweep: false,
            engines,
            settings: EngineSettings::default(),
            machine: Machine::default(),
            workspace,
            tol: 0.02,
            mixed_exx: 1e-3,
            tab: Tab::Cell,
            bundles: Vec::new(),
            progress: Arc::new(Mutex::new(Vec::new())),
            running: None,
            selected_case: None,
            status: "Set the unit cell, choose a solver, then write and run.".into(),
            rejected: Vec::new(),
            active: Engine::Local,
            request_deck: None,
        };
        s.refresh();
        s
    }

    /// container images, pins and addresses of the client engines (set in "Engines & machine")
    pub fn engine_settings(&self) -> EngineSettings {
        self.settings.clone()
    }

    pub fn workspace(&self) -> PathBuf {
        PathBuf::from(&self.workspace)
    }

    fn refresh(&mut self) {
        self.bundles = jobs::scan(&PathBuf::from(&self.workspace));
        if self.selected_case.is_none() {
            self.selected_case = self.bundles.iter().find(|b| b.status == Status::Done).map(|b| b.case_id.clone());
        }
    }

    fn chosen_engines(&self) -> Vec<Engine> {
        let mut v = vec![self.active];
        v.extend(self.engines.iter().filter(|(e, on)| **on && **e != self.active).map(|(e, _)| *e));
        v
    }

    /// the solver selector: reverse-engineered solver and client engines side by side
    fn solver_selector(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).fill(Color32::from_rgb(236, 241, 250)).show(ui, |ui| {
            ui.strong("Solver");
            ui.label(egui::RichText::new("Reverse-engineered").small().color(Color32::DARK_GRAY));
            ui.radio_value(&mut self.active, Engine::Local, Engine::Local.label()).on_hover_text(Engine::Local.role());
            ui.label(egui::RichText::new("Client engines (driven through their own interfaces)").small().color(Color32::DARK_GRAY));
            for e in [Engine::CodeAster, Engine::Fenicsx, Engine::Mapdl] {
                ui.radio_value(&mut self.active, e, e.label()).on_hover_text(e.role());
            }
            ui.label(
                egui::RichText::new(format!("{} · pairing: {} · stress: {}", self.active.role(), self.active.pairing(), self.active.stress_route()))
                    .small(),
            );
            ui.collapsing("Cross-check: also prepare the same job on", |ui| {
                for e in Engine::ALL {
                    if e == self.active {
                        continue;
                    }
                    let on = self.engines.entry(e).or_insert(false);
                    ui.checkbox(on, e.label());
                }
            });
            if self.active == Engine::Local {
                let deck = self
                    .bundles
                    .iter()
                    .find(|b| Some(&b.case_id) == self.selected_case.as_ref() && b.engine == Engine::Local)
                    .and_then(|b| std::fs::read_to_string(b.dir.join("job_local.dat")).ok());
                let label = if deck.is_some() { "Put this case's APDL deck into the code window" } else { "Go to the code window" };
                if ui.button(label).clicked() {
                    self.request_deck = Some(deck.unwrap_or_default());
                }
            }
        });
    }

    fn write_bundles(&mut self) {
        let sweep = Sweep {
            name: if self.use_sweep { self.sweep_name.clone() } else { "single".into() },
            base: self.job.clone(),
            theta_deg: if self.use_sweep { self.theta.clone() } else { Range { from: self.job.cell.theta_deg, to: self.job.cell.theta_deg, steps: 1 } },
            t: if self.use_sweep { self.t.clone() } else { Range { from: self.job.cell.t, to: self.job.cell.t, steps: 1 } },
            engines: self.chosen_engines(),
        };
        let ex = sweep.expand();
        self.rejected = ex.rejected.clone();
        if let Some(j) = ex.jobs.first() {
            self.selected_case = Some(j.case_id());
        }
        let ws = PathBuf::from(&self.workspace);
        let mut n = 0;
        let mut err = None;
        for j in &ex.jobs {
            for &e in &sweep.engines {
                match jobs::write_bundle(&ws, &sweep.name, j, e, &self.settings) {
                    Ok(_) => n += 1,
                    Err(x) => err = Some(x.to_string()),
                }
            }
        }
        let _ = std::fs::write(ws.join(format!("sweep_{}.json", sweep.name)), serde_json::to_string_pretty(&sweep).unwrap_or_default());
        self.status = match err {
            Some(e) => format!("Could not write every bundle: {e}"),
            None => format!(
                "Wrote {n} job bundle(s) for {} case(s) on {} engine(s); {} parameter set(s) rejected by the feasibility rules.",
                ex.jobs.len(),
                sweep.engines.len(),
                ex.rejected.len()
            ),
        };
        self.refresh();
        self.tab = Tab::Jobs;
    }

    fn run_pending(&mut self) {
        if self.running.as_ref().map(|h| !h.is_finished()).unwrap_or(false) {
            return;
        }
        let todo: Vec<Bundle> = self.bundles.iter().filter(|b| b.status != Status::Done).cloned().collect();
        if todo.is_empty() {
            self.status = "Nothing to run: every bundle already has results.".into();
            return;
        }
        {
            let mut p = self.progress.lock().unwrap();
            p.clear();
            for b in &todo {
                p.push((b.dir.clone(), Status::Queued));
            }
        }
        let (m, p) = (self.machine.clone(), self.progress.clone());
        self.status = format!("Running {} bundle(s)…", todo.len());
        self.running = Some(std::thread::spawn(move || jobs::run_all(todo, m, p)));
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if let Some(h) = &self.running {
            if h.is_finished() {
                self.running = None;
                self.refresh();
                self.status = "Run queue finished.".into();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(500));
            }
        }
        egui::Panel::bottom("client_status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let done = self.bundles.iter().filter(|b| b.status == Status::Done).count();
                    ui.label(format!("{} bundles · {} with results · workspace {}", self.bundles.len(), done, self.workspace));
                });
            });
        });
        egui::Panel::left("model").default_size(360.0).resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.model_panel(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Cell, "Unit cell");
                ui.selectable_value(&mut self.tab, Tab::Jobs, "Job bundles");
                ui.selectable_value(&mut self.tab, Tab::Compare, "Cross-engine comparison");
                ui.selectable_value(&mut self.tab, Tab::Verify, "Verification ladder");
                ui.selectable_value(&mut self.tab, Tab::Plan, "Engines & machine");
            });
            ui.separator();
            match self.tab {
                Tab::Cell => self.cell_tab(ui),
                Tab::Jobs => self.jobs_tab(ui),
                Tab::Compare => self.compare_tab(ui),
                Tab::Verify => self.verify_tab(ui),
                Tab::Plan => self.plan_tab(ui),
            }
        });
    }

    fn model_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Neutral model");
        ui.label(egui::RichText::new("Shared by every solver; each adapter translates it.").small());
        ui.add_space(4.0);
        self.solver_selector(ui);
        ui.add_space(6.0);
        ui.strong("Unit cell");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.job.cell.kind, CellKind::RotatingSquares, "Rotating squares");
            ui.selectable_value(&mut self.job.cell.kind, CellKind::Homogeneous, "Homogeneous (check)");
        });
        egui::Grid::new("cell").num_columns(2).show(ui, |ui| {
            ui.label("Square side a");
            ui.add(egui::DragValue::new(&mut self.job.cell.a).speed(0.1).range(0.01..=1e6));
            ui.end_row();
            if self.job.cell.kind == CellKind::RotatingSquares {
                ui.label("Rotation θ (deg)");
                ui.add(egui::DragValue::new(&mut self.job.cell.theta_deg).speed(0.5).range(0.0..=44.9));
                ui.end_row();
                ui.label("Hinge diameter t");
                ui.add(egui::DragValue::new(&mut self.job.cell.t).speed(0.05).range(0.001..=1e6));
                ui.end_row();
            }
            ui.label("Cell side L");
            ui.label(format!("{:.5}", self.job.cell.cell_side()));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.strong("Material (plane stress)");
        egui::Grid::new("mat").num_columns(2).show(ui, |ui| {
            ui.label("Young's modulus E");
            ui.add(egui::DragValue::new(&mut self.job.material.e).speed(10.0).range(1e-9..=1e15));
            ui.end_row();
            ui.label("Poisson's ratio ν");
            ui.add(egui::DragValue::new(&mut self.job.material.nu).speed(0.005).range(-0.99..=0.499));
            ui.end_row();
            ui.label("Thickness");
            ui.add(egui::DragValue::new(&mut self.job.material.thickness).speed(0.05).range(1e-9..=1e6));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.strong("Load case");
        let mut homog = matches!(self.job.load, LoadCase::Homogenize { .. });
        ui.horizontal(|ui| {
            ui.radio_value(&mut homog, true, "Homogenise (3 unit strains)");
            ui.radio_value(&mut homog, false, "One strain state");
        });
        match (&mut self.job.load, homog) {
            (LoadCase::Homogenize { strain }, true) => {
                ui.horizontal(|ui| {
                    ui.label("Strain magnitude");
                    ui.add(egui::DragValue::new(strain).speed(1e-4).range(1e-9..=0.1));
                });
            }
            (LoadCase::Strain { exx, eyy, gxy }, false) => {
                ui.horizontal(|ui| {
                    ui.label("εxx");
                    ui.add(egui::DragValue::new(exx).speed(1e-4));
                    ui.label("εyy");
                    ui.add(egui::DragValue::new(eyy).speed(1e-4));
                    ui.label("γxy");
                    ui.add(egui::DragValue::new(gxy).speed(1e-4));
                });
            }
            (_, true) => self.job.load = LoadCase::Homogenize { strain: 1e-3 },
            (_, false) => self.job.load = LoadCase::Strain { exx: 1e-3, eyy: 0., gxy: 0. },
        }
        ui.label(egui::RichText::new("Mixed strain/stress control is derived from the homogenised tensor (exact in linear elasticity), see Comparison.").small());
        ui.add_space(6.0);
        ui.strong("Mesh (Gmsh)");
        egui::Grid::new("mesh").num_columns(2).show(ui, |ui| {
            ui.label("Element size");
            ui.add(egui::DragValue::new(&mut self.job.mesh.size).speed(0.01).range(1e-6..=1e6));
            ui.end_row();
            ui.label("Element order");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.job.mesh.order, 1, "1");
                ui.selectable_value(&mut self.job.mesh.order, 2, "2");
            });
            ui.end_row();
            ui.label("Mesh seed");
            ui.add(egui::DragValue::new(&mut self.job.mesh.seed));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.checkbox(&mut self.use_sweep, "Parametric sweep");
        if self.use_sweep {
            egui::Grid::new("sweep").num_columns(4).show(ui, |ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut self.sweep_name);
                ui.end_row();
                for (lab, r) in [("θ (deg)", &mut self.theta), ("t", &mut self.t)] {
                    ui.label(lab);
                    ui.add(egui::DragValue::new(&mut r.from).speed(0.05).prefix("from "));
                    ui.add(egui::DragValue::new(&mut r.to).speed(0.05).prefix("to "));
                    ui.add(egui::DragValue::new(&mut r.steps).range(1..=200).suffix(" steps"));
                    ui.end_row();
                }
            });
            let mut probe = Sweep {
                name: self.sweep_name.clone(),
                base: self.job.clone(),
                theta_deg: self.theta.clone(),
                t: self.t.clone(),
                engines: self.chosen_engines(),
            };
            probe.base.cell.kind = self.job.cell.kind;
            let ex = probe.expand();
            ui.label(format!("{} feasible case(s), {} rejected, {} bundle(s)", ex.jobs.len(), ex.rejected.len(), ex.jobs.len() * probe.engines.len()));
        }
        let f = self.job.cell.feasibility(&self.job.mesh);
        for m in &f {
            ui.colored_label(Color32::from_rgb(170, 60, 20), format!("! {m}"));
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let ok = f.is_empty() || self.use_sweep;
            let busy0 = self.running.is_some();
            if ui
                .add_enabled(ok && !busy0, egui::Button::new(egui::RichText::new(format!("Write and run on {}", self.active.label())).strong()))
                .clicked()
            {
                self.write_bundles();
                self.run_pending();
            }
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(ok_bundles(&f, self.use_sweep), egui::Button::new("Write job bundles")).clicked() {
                self.write_bundles();
            }
            let busy = self.running.is_some();
            if ui.add_enabled(!busy, egui::Button::new("Run pending")).clicked() {
                self.run_pending();
            }
            if ui.button("Refresh").clicked() {
                self.refresh();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Workspace");
            ui.add(egui::TextEdit::singleline(&mut self.workspace).desired_width(220.0));
        });
    }

    fn cell_tab(&mut self, ui: &mut egui::Ui) {
        let c = self.job.cell.clone();
        let l = c.cell_side();
        ui.label(format!(
            "L = {:.4}   solid fraction ≈ {:.3}   periodic faces: left/right, bottom/top   physical groups: left, right, bottom, top, solid, anchor{}",
            l,
            c.solid_fraction(),
            if c.kind == CellKind::Homogeneous { ", corner_tr" } else { "" }
        ));
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let r = resp.rect;
        painter.rect_filled(r, 0.0, Color32::from_rgb(246, 247, 250));
        let sc = (r.width().min(r.height()) * 0.42) as f64 / (l * 0.5 * 1.25);
        let ctr = r.center();
        let to = |p: [f64; 2]| Pos2::new(ctr.x + (p[0] * sc) as f32, ctr.y - (p[1] * sc) as f32);
        // neighbouring cells, faint, to show the periodicity
        for di in -1..=1 {
            for dj in -1..=1 {
                let off = [di as f64 * l, dj as f64 * l];
                let faint = di != 0 || dj != 0;
                let fill = if faint { Color32::from_rgb(214, 220, 232) } else { Color32::from_rgb(70, 110, 175) };
                if c.kind == CellKind::Homogeneous {
                    let p0 = to([off[0] - l / 2., off[1] + l / 2.]);
                    let p1 = to([off[0] + l / 2., off[1] - l / 2.]);
                    painter.rect_filled(egui::Rect::from_two_pos(p0, p1), 0.0, fill);
                    continue;
                }
                for (cc, sgn) in c.squares() {
                    let pts: Vec<Pos2> = c.square_corners([cc[0] + off[0], cc[1] + off[1]], sgn).into_iter().map(|p| to(p)).collect();
                    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
                }
                for h in c.hinges() {
                    painter.circle_filled(to([h[0] + off[0], h[1] + off[1]]), (c.t / 2. * sc) as f32, fill);
                }
            }
        }
        let p0 = to([-l / 2., l / 2.]);
        let p1 = to([l / 2., -l / 2.]);
        painter.rect_stroke(egui::Rect::from_two_pos(p0, p1), 0.0, Stroke::new(2.0, Color32::from_rgb(220, 90, 30)), egui::StrokeKind::Middle);
        for (txt, p, al) in [
            ("left", [-l / 2., 0.], egui::Align2::RIGHT_CENTER),
            ("right", [l / 2., 0.], egui::Align2::LEFT_CENTER),
            ("bottom", [0., -l / 2.], egui::Align2::CENTER_TOP),
            ("top", [0., l / 2.], egui::Align2::CENTER_BOTTOM),
        ] {
            let off = match al {
                egui::Align2::RIGHT_CENTER => Vec2::new(-8., 0.),
                egui::Align2::LEFT_CENTER => Vec2::new(8., 0.),
                egui::Align2::CENTER_TOP => Vec2::new(0., 8.),
                _ => Vec2::new(0., -8.),
            };
            painter.text(to(p) + off, al, txt, egui::FontId::proportional(13.0), Color32::from_rgb(200, 80, 20));
        }
        painter.text(
            r.left_top() + Vec2::new(12., 10.),
            egui::Align2::LEFT_TOP,
            "Preview of the cell the Gmsh script builds (faint: periodic neighbours, orange: cell boundary)",
            egui::FontId::proportional(12.0),
            Color32::DARK_GRAY,
        );
    }

    fn jobs_tab(&mut self, ui: &mut egui::Ui) {
        if self.bundles.is_empty() {
            ui.label("No bundles yet. Use \"Write job bundles\".");
        }
        let prog = self.progress.lock().unwrap().clone();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("bundles").striped(true).num_columns(6).show(ui, |ui| {
                ui.strong("Case");
                ui.strong("θ / t");
                ui.strong("Engine");
                ui.strong("Status");
                ui.strong("Folder");
                ui.strong("");
                ui.end_row();
                for b in &self.bundles {
                    let st = prog.iter().find(|p| p.0 == b.dir).map(|p| p.1.clone()).unwrap_or(b.status.clone());
                    ui.label(&b.case_id);
                    ui.label(format!("{:.2}° / {:.3}", b.job.cell.theta_deg, b.job.cell.t));
                    ui.label(b.engine.label());
                    match &st {
                        Status::Done => ui.colored_label(Color32::from_rgb(30, 130, 60), "results"),
                        Status::Failed(f) => ui.colored_label(Color32::from_rgb(180, 50, 30), "failed").on_hover_text(f),
                        Status::Running => ui.colored_label(Color32::from_rgb(30, 90, 170), "running"),
                        Status::Queued => ui.label("queued"),
                        Status::Written => ui.label("written"),
                    };
                    ui.label(egui::RichText::new(b.dir.display().to_string()).small());
                    if ui.small_button("Compare").clicked() {
                        self.selected_case = Some(b.case_id.clone());
                        self.tab = Tab::Compare;
                    }
                    ui.end_row();
                }
            });
            if !self.rejected.is_empty() {
                ui.add_space(8.0);
                ui.strong("Rejected by the feasibility rules");
                for (k, why) in &self.rejected {
                    ui.label(format!("{k}: {}", why.join("; ")));
                }
            }
        });
    }

    fn case_results(&self) -> (Option<Job>, Vec<Effective>) {
        let Some(id) = &self.selected_case else { return (None, vec![]) };
        let mut job = None;
        let mut effs = Vec::new();
        for b in self.bundles.iter().filter(|b| &b.case_id == id) {
            job = Some(b.job.clone());
            if let Some(e) = compare::effective(b) {
                effs.push(e);
            }
        }
        (job, effs)
    }

    fn case_picker(&mut self, ui: &mut egui::Ui) {
        let mut ids: Vec<String> = self.bundles.iter().map(|b| b.case_id.clone()).collect();
        ids.dedup();
        ui.horizontal(|ui| {
            ui.label("Case");
            egui::ComboBox::from_id_salt("case").selected_text(self.selected_case.clone().unwrap_or_else(|| "—".into())).show_ui(ui, |ui| {
                for id in ids {
                    let b = self.bundles.iter().find(|b| b.case_id == id).unwrap();
                    let lab = format!("{id}  θ {:.2}°  t {:.3}", b.job.cell.theta_deg, b.job.cell.t);
                    ui.selectable_value(&mut self.selected_case, Some(id.clone()), lab);
                }
            });
            ui.label("Tolerance");
            ui.add(egui::DragValue::new(&mut self.tol).speed(0.001).range(0.0..=1.0).custom_formatter(|v, _| format!("{:.1} %", v * 100.)));
        });
    }

    fn compare_tab(&mut self, ui: &mut egui::Ui) {
        self.case_picker(ui);
        let (_, effs) = self.case_results();
        if effs.is_empty() {
            ui.label("No engine has produced results for this case yet. Results appear after a run (results.json in each bundle).");
            return;
        }
        let rows = compare::cross_check(&effs, self.tol);
        let engines: Vec<Engine> = effs.iter().map(|e| e.engine).collect();
        egui::Grid::new("cmp").striped(true).show(ui, |ui| {
            ui.strong("Quantity");
            for e in &engines {
                ui.strong(e.label());
            }
            ui.strong("Max relative difference");
            ui.end_row();
            for r in &rows {
                ui.label(&r.name);
                for e in &engines {
                    ui.monospace(r.values.get(e).map(|v| format!("{v:.6e}")).unwrap_or_default());
                }
                let t = format!("{:.3} %", r.max_rel_diff * 100.);
                if r.flagged {
                    ui.colored_label(Color32::from_rgb(190, 40, 20), format!("{t}  exceeds tolerance"));
                } else {
                    ui.label(t);
                }
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.strong("Mixed control: prescribed εxx, free lateral stress (σyy = τxy = 0)");
        ui.horizontal(|ui| {
            ui.label("εxx");
            ui.add(egui::DragValue::new(&mut self.mixed_exx).speed(1e-4));
        });
        for e in &effs {
            if e.complete {
                let (eyy, gxy, sxx) = compare::uniaxial_response(e, self.mixed_exx);
                ui.label(format!("{}: εyy = {:.4e}, γxy = {:.4e}, σxx = {:.4e}", e.engine.label(), eyy, gxy, sxx));
            }
        }
        ui.add_space(8.0);
        for e in &effs {
            ui.label(egui::RichText::new(format!("{}: version {}, pairing: {}, stress: {}", e.engine.label(), e.engine_version, e.engine.pairing(), e.engine.stress_route())).small());
        }
    }

    fn verify_tab(&mut self, ui: &mut egui::Ui) {
        self.case_picker(ui);
        let (job, effs) = self.case_results();
        ui.label("Steps 1–3 are judged per engine from its own result; step 4 is the cross-engine comparison. These are verification only — validation needs an experiment.");
        let Some(job) = job else { return };
        for e in &effs {
            ui.add_space(6.0);
            ui.strong(e.engine.label());
            for c in compare::ladder(&job, e, self.tol) {
                let (txt, col) = match c.passed {
                    Some(true) => ("pass", Color32::from_rgb(30, 130, 60)),
                    Some(false) => ("FAIL", Color32::from_rgb(190, 40, 20)),
                    None => ("trend", Color32::from_rgb(120, 120, 120)),
                };
                ui.horizontal(|ui| {
                    ui.colored_label(col, txt);
                    ui.label(format!("[{}] {} — {}", c.step, c.name, c.detail));
                });
            }
        }
        for b in self.bundles.iter().filter(|b| Some(&b.case_id) == self.selected_case.as_ref()) {
            if let Ok(t) = std::fs::read_to_string(b.dir.join("mesh_check.json")) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                    ui.label(format!(
                        "[3 consistency] mesh conversion for {} ({}): nodes {} → {}, area {} → {}, missing groups {}",
                        b.engine.label(),
                        v["format"],
                        v["nodes_in"],
                        v["nodes_out"],
                        v["area_in"],
                        v["area_out"],
                        v["groups_missing"]
                    ));
                }
            }
        }
        if effs.len() >= 2 {
            let rows = compare::cross_check(&effs, self.tol);
            let bad = rows.iter().filter(|r| r.flagged).count();
            ui.add_space(6.0);
            ui.label(format!("[4 cross-code] {} engines, {} of {} quantities beyond {:.1} %", effs.len(), bad, rows.len(), self.tol * 100.));
        }
    }

    fn plan_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("Engines");
            egui::Grid::new("eng").striped(true).show(ui, |ui| {
                ui.strong("Engine");
                ui.strong("Role");
                ui.strong("Mesh format");
                ui.strong("Periodic pairing");
                ui.strong("Effective stress");
                ui.end_row();
                for e in Engine::ALL {
                    ui.label(e.label());
                    ui.label(e.role());
                    ui.label(e.mesh_format());
                    ui.label(e.pairing());
                    ui.label(e.stress_route());
                    ui.end_row();
                }
            });
            ui.add_space(8.0);
            ui.heading("Containers and pins");
            egui::Grid::new("set").num_columns(2).show(ui, |ui| {
                let s = &mut self.settings;
                for (lab, v) in [
                    ("Container runtime", &mut s.runtime),
                    ("Code_Aster image", &mut s.aster_image),
                    ("FEniCSx + dolfinx_mpc image", &mut s.fenicsx_image),
                    ("MAPDL container address", &mut s.mapdl_ip),
                    ("gmsh (uv)", &mut s.gmsh_pin),
                    ("meshlane (uv)", &mut s.meshlane_pin),
                    ("PyMAPDL (uv)", &mut s.pymapdl_pin),
                ] {
                    ui.label(lab);
                    ui.add(egui::TextEdit::singleline(v).desired_width(320.0));
                    ui.end_row();
                }
                ui.label("MAPDL gRPC port");
                ui.add(egui::DragValue::new(&mut s.mapdl_port));
                ui.end_row();
            });
            if !self.settings.meshlane_pin.contains("==") {
                ui.colored_label(Color32::from_rgb(170, 90, 0), "meshlane is not pinned: write an exact version (meshlane==x.y.z). The version actually used is recorded in each bundle's mesh_check.json.");
            }
            ui.add_space(8.0);
            ui.heading("Machine limits");
            egui::Grid::new("mach").num_columns(2).show(ui, |ui| {
                ui.label("RAM budget for engines (MB)");
                ui.add(egui::DragValue::new(&mut self.machine.ram_budget_mb).speed(64));
                ui.end_row();
                for e in Engine::ALL {
                    ui.label(format!("RAM per {} run (MB)", e.label()));
                    ui.add(egui::DragValue::new(self.machine.ram_per_engine_mb.entry(e).or_insert(2000)).speed(64));
                    ui.end_row();
                }
                ui.label("MAPDL licence seats");
                ui.add(egui::DragValue::new(&mut self.machine.mapdl_seats).range(1..=64));
                ui.end_row();
                ui.label("Cores");
                ui.add(egui::DragValue::new(&mut self.machine.cores).range(1..=256));
                ui.end_row();
            });
            for e in Engine::ALL {
                ui.label(format!("{}: at most {} run(s) at once", e.label(), self.machine.max_parallel(e)));
            }
            ui.add_space(8.0);
            ui.heading("Each bundle contains");
            ui.label("params.json (neutral model), geometry.py (Gmsh, shared), convert_mesh.py (meshlane + round-trip check), the engine's own job file, run.sh, and bundle.json with engine image, pins, mesh size, order, seed and SHA-256 of every input. After a run: geometry.json, mesh_check.json, results.json, normalized.json, log.txt.");
        });
    }
}

