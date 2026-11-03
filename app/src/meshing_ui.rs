//! The Meshing tab: buttons that write geometry, mesh, materials, conditions and loads into the
//! code window (for quick hand adjustments). Every button edits the code text directly.

use crate::meshing::{self, fmt, on_plane, Section};
use eframe::egui;
use egui::Color32;
use std::path::PathBuf;

const ETS: [(&str, &str); 7] = [
    ("SOLID185", "8-node brick"),
    ("SOLID186", "20-node brick"),
    ("SOLID187", "10-node tetrahedron"),
    ("PLANE182", "4-node quad / triangle"),
    ("PLANE183", "8-node quad"),
    ("BEAM188", "3-D beam"),
    ("LINK180", "bar"),
];
const PLANE_OPTS: [&str; 4] = ["plane stress", "axisymmetric", "plane strain", "plane stress with thickness"];
const DOFS: [&str; 7] = ["ALL", "UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"];
const DIRS: [&str; 3] = ["FX", "FY", "FZ"];

pub struct MeshForm {
    title: String,
    et: usize,
    et_id: usize,
    plane_opt: usize,
    thick: f64,
    mat: usize,
    e: f64,
    nu: f64,
    dens: f64,
    beam_b: f64,
    beam_h: f64,
    link_a: f64,
    block: [f64; 6],
    rect: [f64; 4],
    esize: f64,
    node: (usize, [f64; 3]),
    elem: String,
    line_a: [f64; 3],
    line_b: [f64; 3],
    line_n: usize,
    line_first: usize,
    fix_axis: usize,
    fix_at: f64,
    fix_dof: usize,
    dn: (usize, usize, f64),
    fp_axis: usize,
    fp_at: f64,
    fp_dir: usize,
    fp_total: f64,
    fnode: (usize, usize, f64),
    pr_axis: usize,
    pr_at: f64,
    pr_p: f64,
    grav: f64,
    grav_axis: usize,
    eq_pcg: bool,
    eq_tol: f64,
    cell_a: f64,
    cell_theta: f64,
    cell_t: f64,
    cell_h: f64,
    cell_order: u8,
    undo: Vec<String>,
    pub last: String,
}

impl Default for MeshForm {
    fn default() -> Self {
        MeshForm {
            title: "New model".into(),
            et: 0,
            et_id: 1,
            plane_opt: 0,
            thick: 1.0,
            mat: 1,
            e: 210000.0,
            nu: 0.3,
            dens: 7.85e-9,
            beam_b: 20.0,
            beam_h: 40.0,
            link_a: 100.0,
            block: [0., 100., 0., 20., 0., 20.],
            rect: [0., 100., 0., 20.],
            esize: 5.0,
            node: (1, [0.; 3]),
            elem: "1,2".into(),
            line_a: [0.; 3],
            line_b: [1000., 0., 0.],
            line_n: 10,
            line_first: 1,
            fix_axis: 0,
            fix_at: 0.0,
            fix_dof: 0,
            dn: (1, 1, 0.0),
            fp_axis: 0,
            fp_at: 100.0,
            fp_dir: 1,
            fp_total: -1000.0,
            fnode: (1, 1, -100.0),
            pr_axis: 1,
            pr_at: 20.0,
            pr_p: 1.0,
            grav: 9810.0,
            grav_axis: 1,
            eq_pcg: false,
            eq_tol: 1.0e-8,
            cell_a: 10.0,
            cell_theta: 15.0,
            cell_t: 1.0,
            cell_h: 0.5,
            cell_order: 2,
            undo: Vec::new(),
            last: String::new(),
        }
    }
}

fn dv(ui: &mut egui::Ui, v: &mut f64) {
    ui.add(egui::DragValue::new(v).speed(0.1).max_decimals(6));
}

fn axis_pick(ui: &mut egui::Ui, id: &str, a: &mut usize) {
    egui::ComboBox::from_id_salt(id).width(40.0).selected_text(meshing::axis_name(*a)).show_ui(ui, |ui| {
        for k in 0..3 {
            ui.selectable_value(a, k, meshing::axis_name(k));
        }
    });
}

fn pick(ui: &mut egui::Ui, id: &str, v: &mut usize, items: &[&str]) {
    egui::ComboBox::from_id_salt(id).width(70.0).selected_text(items[*v]).show_ui(ui, |ui| {
        for (k, it) in items.iter().enumerate() {
            ui.selectable_value(v, k, *it);
        }
    });
}

impl MeshForm {
    fn put(&mut self, code: &mut String, sec: Section, snippet: &str, what: &str) {
        self.undo.push(code.clone());
        let line = meshing::insert(code, sec, snippet);
        self.last = format!("{what}: inserted at line {} of the code.", line + 1);
    }

    /// draws the tab; returns true when the code changed
    pub fn ui(&mut self, ui: &mut egui::Ui, code: &mut String, workspace: PathBuf) -> bool {
        let before = code.clone();
        ui.label("Each button writes MAPDL commands into the code window, in the right section (geometry and materials into /PREP7, conditions and loads before SOLVE). Edit the code by hand afterwards if needed.");
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo last change")).clicked() {
                if let Some(prev) = self.undo.pop() {
                    *code = prev;
                    self.last = "Last change undone.".into();
                }
            }
            if !self.last.is_empty() {
                ui.colored_label(Color32::from_rgb(30, 110, 60), &self.last);
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(2, |cols| {
                // ------------------------------------------------ left column: model, geometry, mesh
                let ui = &mut cols[0];
                ui.heading("Model");
                ui.horizontal(|ui| {
                    ui.label("Title");
                    ui.text_edit_singleline(&mut self.title);
                    if ui.button("Set").clicked() {
                        self.undo.push(code.clone());
                        meshing::set_title(code, &self.title);
                        self.last = "Title set.".into();
                    }
                });
                if ui.button("Start a new empty model (replaces the code; Undo brings it back)").clicked() {
                    self.undo.push(code.clone());
                    *code = meshing::SKELETON.replace("New model", &self.title);
                    self.last = "New model skeleton written.".into();
                }
                ui.add_space(6.0);
                ui.heading("Element type and material");
                ui.horizontal(|ui| {
                    ui.label("Type");
                    ui.add(egui::DragValue::new(&mut self.et_id).range(1..=999).prefix("#"));
                    egui::ComboBox::from_id_salt("et").selected_text(format!("{} ({})", ETS[self.et].0, ETS[self.et].1)).show_ui(ui, |ui| {
                        for (k, (n, d)) in ETS.iter().enumerate() {
                            ui.selectable_value(&mut self.et, k, format!("{n} ({d})"));
                        }
                    });
                });
                let plane = ETS[self.et].0.starts_with("PLANE");
                if plane {
                    ui.horizontal(|ui| {
                        pick(ui, "popt", &mut self.plane_opt, &PLANE_OPTS);
                        if self.plane_opt == 3 {
                            ui.label("thickness");
                            dv(ui, &mut self.thick);
                        }
                    });
                }
                if ui.button("Insert element type").clicked() {
                    let mut s = format!("ET,{},{}", self.et_id, ETS[self.et].0);
                    if plane && self.plane_opt > 0 {
                        s += &format!("\nKEYOPT,{},3,{}", self.et_id, self.plane_opt);
                    }
                    if plane && self.plane_opt == 3 {
                        s += &format!("\nR,{},{}", self.et_id, fmt(self.thick));
                    }
                    s += &format!("\nTYPE,{}", self.et_id);
                    self.put(code, Section::Prep, &s, "Element type");
                }
                ui.horizontal(|ui| {
                    ui.label("Material");
                    ui.add(egui::DragValue::new(&mut self.mat).range(1..=999).prefix("#"));
                    ui.label("E");
                    dv(ui, &mut self.e);
                    ui.label("ν");
                    ui.add(egui::DragValue::new(&mut self.nu).speed(0.005).range(-0.99..=0.499));
                    ui.label("ρ");
                    ui.add(egui::DragValue::new(&mut self.dens).speed(1e-10).max_decimals(12));
                });
                if ui.button("Insert material").clicked() {
                    let s = format!("MP,EX,{m},{}\nMP,PRXY,{m},{}\nMP,DENS,{m},{}\nMAT,{m}", fmt(self.e), self.nu, self.dens, m = self.mat);
                    self.put(code, Section::Prep, &s, "Material");
                }
                ui.horizontal(|ui| {
                    ui.label("Beam rectangle B × H");
                    dv(ui, &mut self.beam_b);
                    dv(ui, &mut self.beam_h);
                    if ui.button("Insert beam section").clicked() {
                        let s = format!("SECTYPE,1,BEAM,RECT\nSECDATA,{},{}", fmt(self.beam_b), fmt(self.beam_h));
                        self.put(code, Section::Prep, &s, "Beam section");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Bar area");
                    dv(ui, &mut self.link_a);
                    if ui.button("Insert bar section").clicked() {
                        let s = format!("SECTYPE,1,LINK\nSECDATA,{}", fmt(self.link_a));
                        self.put(code, Section::Prep, &s, "Bar section");
                    }
                });
                ui.add_space(6.0);
                ui.heading("Geometry");
                ui.horizontal(|ui| {
                    ui.label("Block x");
                    dv(ui, &mut self.block[0]);
                    dv(ui, &mut self.block[1]);
                    ui.label("y");
                    dv(ui, &mut self.block[2]);
                    dv(ui, &mut self.block[3]);
                    ui.label("z");
                    dv(ui, &mut self.block[4]);
                    dv(ui, &mut self.block[5]);
                });
                if ui.button("Insert block (BLOCK)").clicked() {
                    let b = self.block;
                    let s = format!("BLOCK,{},{},{},{},{},{}", fmt(b[0]), fmt(b[1]), fmt(b[2]), fmt(b[3]), fmt(b[4]), fmt(b[5]));
                    self.put(code, Section::Prep, &s, "Block");
                }
                ui.horizontal(|ui| {
                    ui.label("Rectangle x");
                    dv(ui, &mut self.rect[0]);
                    dv(ui, &mut self.rect[1]);
                    ui.label("y");
                    dv(ui, &mut self.rect[2]);
                    dv(ui, &mut self.rect[3]);
                    if ui.button("Insert rectangle (RECTNG)").clicked() {
                        let r = self.rect;
                        let s = format!("RECTNG,{},{},{},{}", fmt(r[0]), fmt(r[1]), fmt(r[2]), fmt(r[3]));
                        self.put(code, Section::Prep, &s, "Rectangle");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Node #");
                    ui.add(egui::DragValue::new(&mut self.node.0).range(1..=100_000_000));
                    for k in 0..3 {
                        dv(ui, &mut self.node.1[k]);
                    }
                    if ui.button("Insert node (N)").clicked() {
                        let (n, x) = self.node;
                        let s = format!("N,{n},{},{},{}", fmt(x[0]), fmt(x[1]), fmt(x[2]));
                        self.node.0 += 1;
                        self.put(code, Section::Prep, &s, "Node");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Element nodes");
                    ui.add(egui::TextEdit::singleline(&mut self.elem).desired_width(160.0));
                    if ui.button("Insert element (E)").clicked() {
                        let s = format!("E,{}", self.elem.replace(' ', ","));
                        self.put(code, Section::Prep, &s, "Element");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Line of beam/bar elements from");
                    for k in 0..3 {
                        dv(ui, &mut self.line_a[k]);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("to");
                    for k in 0..3 {
                        dv(ui, &mut self.line_b[k]);
                    }
                    ui.label("elements");
                    ui.add(egui::DragValue::new(&mut self.line_n).range(1..=100000));
                    ui.label("first node #");
                    ui.add(egui::DragValue::new(&mut self.line_first).range(1..=100_000_000));
                });
                if ui.button("Insert line of nodes and elements").clicked() {
                    let (a, b, n, f) = (self.line_a, self.line_b, self.line_n, self.line_first);
                    let s = format!(
                        "! line of {n} elements from ({},{},{}) to ({},{},{})\n*DO,i,0,{n}\n  N,{f}+i,{}+i*({})/{n},{}+i*({})/{n},{}+i*({})/{n}\n*ENDDO\n*DO,i,0,{}\n  E,{f}+i,{f}+i+1\n*ENDDO",
                        fmt(a[0]), fmt(a[1]), fmt(a[2]), fmt(b[0]), fmt(b[1]), fmt(b[2]),
                        fmt(a[0]), fmt(b[0] - a[0]), fmt(a[1]), fmt(b[1] - a[1]), fmt(a[2]), fmt(b[2] - a[2]), n - 1
                    );
                    self.line_first += n + 1;
                    self.put(code, Section::Prep, &s, "Line of elements");
                }
                ui.add_space(6.0);
                ui.heading("Mesh");
                ui.horizontal(|ui| {
                    ui.label("Element size");
                    dv(ui, &mut self.esize);
                    if ui.button("ESIZE").clicked() {
                        let s = format!("ESIZE,{}", fmt(self.esize));
                        self.put(code, Section::Prep, &s, "Element size");
                    }
                    if ui.button("Mesh volumes (VMESH)").clicked() {
                        self.put(code, Section::Prep, "VMESH,ALL", "Volume mesh");
                    }
                    if ui.button("Mesh areas (AMESH)").clicked() {
                        self.put(code, Section::Prep, "AMESH,ALL", "Area mesh");
                    }
                    if ui.button("Merge coincident nodes").clicked() {
                        self.put(code, Section::Prep, "NUMMRG,NODE", "Node merge");
                    }
                });
                ui.add_space(6.0);
                ui.heading("Periodic unit cell (Gmsh)");
                ui.horizontal(|ui| {
                    ui.label("a");
                    dv(ui, &mut self.cell_a);
                    ui.label("θ°");
                    ui.add(egui::DragValue::new(&mut self.cell_theta).speed(0.5).range(0.0..=44.9));
                    ui.label("t");
                    dv(ui, &mut self.cell_t);
                    ui.label("size");
                    dv(ui, &mut self.cell_h);
                    ui.selectable_value(&mut self.cell_order, 1, "linear");
                    ui.selectable_value(&mut self.cell_order, 2, "quadratic");
                });
                if ui.button("Write the rotating-squares cell with periodic CEs (replaces the code)").clicked() {
                    let mut job = crate::client::default_job();
                    job.cell.a = self.cell_a;
                    job.cell.theta_deg = self.cell_theta;
                    job.cell.t = self.cell_t;
                    job.mesh.size = self.cell_h;
                    job.mesh.order = self.cell_order;
                    job.material.e = self.e;
                    job.material.nu = self.nu;
                    let f = job.cell.feasibility(&job.mesh);
                    if !f.is_empty() {
                        self.last = format!("Not written: {}", f.join("; "));
                    } else {
                        let r = crate::client::jobs::write_bundle(&workspace, "code_window_cells", &job, crate::client::adapters::Engine::Local, &Default::default())
                            .map_err(|e| e.to_string())
                            .and_then(|b| {
                                let mut log = String::new();
                                crate::client::local::make_deck(&b.dir, &mut log)
                            });
                        match r {
                            Ok(d) => {
                                self.undo.push(code.clone());
                                *code = d.text;
                                self.last = format!("Unit-cell deck written ({} strain cases, {} + {} periodic pairs).", d.cases.len(), d.npr, d.npt);
                            }
                            Err(e) => self.last = format!("Could not build the cell: {e}"),
                        }
                    }
                }
                // ------------------------------------------------ right column: conditions, loads, solution, output
                let ui = &mut cols[1];
                ui.heading("Boundary conditions");
                ui.horizontal(|ui| {
                    ui.label("Fix");
                    pick(ui, "fixdof", &mut self.fix_dof, &DOFS);
                    ui.label("of all nodes at");
                    axis_pick(ui, "fixax", &mut self.fix_axis);
                    ui.label("=");
                    dv(ui, &mut self.fix_at);
                    if ui.button("Insert").clicked() {
                        let s = on_plane(self.fix_axis, self.fix_at, &format!("D,ALL,{},0", DOFS[self.fix_dof]));
                        self.put(code, Section::Solu, &s, "Fixed support");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Displacement at node #");
                    ui.add(egui::DragValue::new(&mut self.dn.0).range(1..=100_000_000));
                    pick(ui, "dndof", &mut self.dn.1, &DOFS);
                    ui.label("=");
                    dv(ui, &mut self.dn.2);
                    if ui.button("Insert").clicked() {
                        let s = format!("D,{},{},{}", self.dn.0, DOFS[self.dn.1], fmt(self.dn.2));
                        self.put(code, Section::Solu, &s, "Nodal displacement");
                    }
                });
                ui.add_space(6.0);
                ui.heading("Loads");
                ui.horizontal(|ui| {
                    ui.label("Total force");
                    pick(ui, "fpdir", &mut self.fp_dir, &DIRS);
                    dv(ui, &mut self.fp_total);
                    ui.label("shared by the nodes at");
                    axis_pick(ui, "fpax", &mut self.fp_axis);
                    ui.label("=");
                    dv(ui, &mut self.fp_at);
                    if ui.button("Insert").clicked() {
                        let s = on_plane(
                            self.fp_axis,
                            self.fp_at,
                            &format!("*GET,nload,NODE,0,COUNT\nF,ALL,{},{}/nload", DIRS[self.fp_dir], fmt(self.fp_total)),
                        );
                        self.put(code, Section::Solu, &s, "Distributed force");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Force at node #");
                    ui.add(egui::DragValue::new(&mut self.fnode.0).range(1..=100_000_000));
                    pick(ui, "fndir", &mut self.fnode.1, &DIRS);
                    dv(ui, &mut self.fnode.2);
                    if ui.button("Insert").clicked() {
                        let s = format!("F,{},{},{}", self.fnode.0, DIRS[self.fnode.1], fmt(self.fnode.2));
                        self.put(code, Section::Solu, &s, "Nodal force");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Pressure");
                    dv(ui, &mut self.pr_p);
                    ui.label("on the face at");
                    axis_pick(ui, "prax", &mut self.pr_axis);
                    ui.label("=");
                    dv(ui, &mut self.pr_at);
                    if ui.button("Insert").clicked() {
                        let s = on_plane(self.pr_axis, self.pr_at, &format!("SF,ALL,PRES,{}", fmt(self.pr_p)));
                        self.put(code, Section::Solu, &s, "Pressure");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Gravity");
                    dv(ui, &mut self.grav);
                    ui.label("acting along −");
                    axis_pick(ui, "gax", &mut self.grav_axis);
                    if ui.button("Insert").clicked() {
                        let mut a = [0.0; 3];
                        a[self.grav_axis] = self.grav; // ACEL is the structure's acceleration: gravity acts opposite
                        let s = format!("ACEL,{},{},{}", fmt(a[0]), fmt(a[1]), fmt(a[2]));
                        self.put(code, Section::Solu, &s, "Gravity");
                    }
                });
                ui.add_space(6.0);
                ui.heading("Solution");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.eq_pcg, "PCG iterative solver");
                    if self.eq_pcg {
                        ui.label("tolerance");
                        ui.add(egui::DragValue::new(&mut self.eq_tol).speed(1e-9).max_decimals(12));
                    }
                    if ui.button("Insert EQSLV").clicked() {
                        let s = if self.eq_pcg { format!("EQSLV,PCG,{:e}", self.eq_tol) } else { "EQSLV,SPARSE".to_string() };
                        self.put(code, Section::Solu, &s, "Equation solver");
                    }
                });
                ui.add_space(6.0);
                ui.heading("Output listings");
                ui.horizontal_wrapped(|ui| {
                    for (lab, cmd) in [
                        ("Displacements", "PRNSOL,U"),
                        ("Stresses", "PRNSOL,S"),
                        ("Principal stresses", "PRNSOL,S,PRIN"),
                        ("Reactions", "PRRSOL"),
                        ("Element forces", "PRESOL"),
                        ("Force sum", "FSUM"),
                    ] {
                        if ui.button(lab).clicked() {
                            self.put(code, Section::Post, cmd, lab);
                        }
                    }
                });
            });
        });
        *code != before
    }
}
