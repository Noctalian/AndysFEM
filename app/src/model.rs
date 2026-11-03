//! The finite element database: nodes, elements, types, materials, loads, selection.

use std::collections::{BTreeMap, HashMap};

pub const DOF_NAMES: [&str; 6] = ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"];
pub const FORCE_NAMES: [&str; 6] = ["FX", "FY", "FZ", "MX", "MY", "MZ"];

pub const M_UXYZ: u8 = 0b000111;
pub const M_UXY: u8 = 0b000011;
pub const M_ALL6: u8 = 0b111111;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Link,
    Beam,
    Spring,
    Plane182,
    Plane183,
    Solid185,
    Solid186,
    Solid187,
    Surf154,
    Mass21,
}

impl Kind {
    pub fn from_name(n: &str) -> Option<(Kind, &'static str)> {
        let n = n.trim().to_uppercase();
        let num = n.trim_start_matches(|c: char| c.is_alphabetic());
        Some(match num {
            "180" | "8" | "1" => (Kind::Link, "LINK180"),
            "188" | "4" => (Kind::Beam, "BEAM188"),
            "14" => (Kind::Spring, "COMBIN14"),
            "182" | "42" => (Kind::Plane182, "PLANE182"),
            "183" | "82" => (Kind::Plane183, "PLANE183"),
            "185" | "45" => (Kind::Solid185, "SOLID185"),
            "186" | "95" => (Kind::Solid186, "SOLID186"),
            "187" | "92" => (Kind::Solid187, "SOLID187"),
            "154" => (Kind::Surf154, "SURF154"),
            "21" => (Kind::Mass21, "MASS21"),
            _ => return None,
        })
    }
    pub fn dof_mask(self) -> u8 {
        match self {
            Kind::Beam => M_ALL6,
            Kind::Plane182 | Kind::Plane183 => M_UXY,
            Kind::Surf154 => 0,
            _ => M_UXYZ,
        }
    }
    pub fn is_solid(self) -> bool {
        matches!(self, Kind::Solid185 | Kind::Solid186 | Kind::Solid187)
    }
    pub fn is_plane(self) -> bool {
        matches!(self, Kind::Plane182 | Kind::Plane183)
    }
}

#[derive(Debug, Clone)]
pub struct EType {
    pub name: String,
    pub kind: Option<Kind>,
    pub keyopt: [i32; 19],
}

#[derive(Debug, Clone)]
pub struct Element {
    pub etype: usize,
    pub mat: usize,
    pub real: usize,
    pub sec: usize,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Section {
    pub kind: String,    // BEAM, LINK, SHELL
    pub subtype: String, // RECT, CSOLID, CTUBE, ASEC
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct Volume {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub is_area: bool,
    pub meshed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ce {
    /// equation number (CE,NEQN; NEXT = highest + 1)
    pub n: usize,
    /// dependent (first) term — the original eliminates the first DOF of each equation
    pub slave: usize,
    /// independent term; 0 = single-term equation (the dependent DOF is prescribed to c)
    pub master: usize,
    pub dof: u8,
    pub c: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Model {
    pub nodes: Vec<Option<[f64; 3]>>,
    pub elems: Vec<Option<Element>>,
    pub etypes: BTreeMap<usize, EType>,
    pub mats: BTreeMap<usize, HashMap<String, f64>>,
    pub reals: BTreeMap<usize, Vec<f64>>,
    pub sections: BTreeMap<usize, Section>,
    pub nsel: Vec<bool>,
    pub esel: Vec<bool>,
    pub comps: HashMap<String, (bool, Vec<usize>)>, // (is_node, ids)
    pub disp: BTreeMap<(usize, u8), f64>,
    pub force: BTreeMap<(usize, u8), f64>,
    pub node_pres: BTreeMap<usize, f64>,
    pub elem_pres: BTreeMap<(usize, usize), f64>, // (elem, face) -> pressure
    pub acel: [f64; 3],
    /// ERESX,NO: copy integration-point stresses to nodes instead of extrapolating
    pub eresx_copy: bool,
    /// two-term constraint equations u(slave) - u(master) = c (APDL CE with coefficients +1/-1)
    pub ces: Vec<Ce>,
    pub volumes: Vec<Volume>,
    pub esize: f64,
    pub ndiv: usize,
    // attributes for new elements
    pub cur_type: usize,
    pub cur_mat: usize,
    pub cur_real: usize,
    pub cur_sec: usize,
}

impl Model {
    pub fn new() -> Self {
        Model {
            cur_type: 1,
            cur_mat: 1,
            cur_real: 1,
            cur_sec: 1,
            ..Default::default()
        }
    }

    pub fn set_node(&mut self, id: usize, xyz: [f64; 3]) {
        if id >= self.nodes.len() {
            self.nodes.resize(id + 1, None);
            self.nsel.resize(id + 1, false);
        }
        self.nodes[id] = Some(xyz);
        self.nsel[id] = true;
    }
    pub fn node(&self, id: usize) -> Option<[f64; 3]> {
        self.nodes.get(id).copied().flatten()
    }
    pub fn set_elem(&mut self, id: usize, e: Element) {
        if id >= self.elems.len() {
            self.elems.resize(id + 1, None);
            self.esel.resize(id + 1, false);
        }
        self.elems[id] = Some(e);
        self.esel[id] = true;
    }
    pub fn max_node(&self) -> usize {
        self.nodes.iter().rposition(|n| n.is_some()).unwrap_or(0)
    }
    pub fn max_elem(&self) -> usize {
        self.elems.iter().rposition(|n| n.is_some()).unwrap_or(0)
    }
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }
    pub fn elem_count(&self) -> usize {
        self.elems.iter().filter(|n| n.is_some()).count()
    }
    pub fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes.iter().enumerate().filter(|(_, n)| n.is_some()).map(|(i, _)| i)
    }
    pub fn sel_node_ids(&self) -> Vec<usize> {
        self.node_ids().filter(|&i| self.nsel[i]).collect()
    }
    pub fn elem_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.elems.iter().enumerate().filter(|(_, n)| n.is_some()).map(|(i, _)| i)
    }
    pub fn sel_elem_ids(&self) -> Vec<usize> {
        self.elem_ids().filter(|&i| self.esel[i]).collect()
    }
    pub fn mat_prop(&self, mat: usize, name: &str) -> Option<f64> {
        self.mats.get(&mat).and_then(|m| m.get(name).copied())
    }
    pub fn kind_of(&self, e: &Element) -> Option<Kind> {
        self.etypes.get(&e.etype).and_then(|t| t.kind)
    }
    pub fn keyopt(&self, e: &Element, k: usize) -> i32 {
        self.etypes.get(&e.etype).map(|t| t.keyopt[k]).unwrap_or(0)
    }
    pub fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for n in self.nodes.iter().flatten() {
            for k in 0..3 {
                lo[k] = lo[k].min(n[k]);
                hi[k] = hi[k].max(n[k]);
            }
        }
        if lo[0] > hi[0] {
            return ([0.0; 3], [1.0; 3]);
        }
        (lo, hi)
    }
}
