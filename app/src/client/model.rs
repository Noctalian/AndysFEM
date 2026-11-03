//! fe-model: the neutral model shared by every engine (geometry parameters, material, load case,
//! mesh settings). Engines never see each other's formats; each adapter translates from this.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellKind {
    /// rotating-squares auxetic cell: 2 x 2 squares rotated +/- theta, joined by circular hinges
    RotatingSquares,
    /// solid square: verification ladder step 1 (must return the material's own stiffness)
    Homogeneous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellParams {
    pub kind: CellKind,
    /// square side
    pub a: f64,
    /// rotation of the squares in degrees (0 = squares meet at corners in a straight line)
    pub theta_deg: f64,
    /// hinge (ligament) diameter
    pub t: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub e: f64,
    pub nu: f64,
    /// out-of-plane thickness (plane stress)
    pub thickness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSettings {
    pub size: f64,
    pub order: u8,
    pub seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoadCase {
    /// three unit macroscopic strains (xx, yy, engineering xy) of magnitude `strain` -> full 2-D tensor
    Homogenize { strain: f64 },
    /// one prescribed macroscopic strain state
    Strain { exx: f64, eyy: f64, gxy: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub cell: CellParams,
    pub material: Material,
    pub mesh: MeshSettings,
    pub load: LoadCase,
}

impl CellParams {
    /// cell side: the squares touch exactly when the centre spacing is a (cos t + sin t)
    pub fn cell_side(&self) -> f64 {
        match self.kind {
            CellKind::Homogeneous => 2.0 * self.a,
            CellKind::RotatingSquares => {
                let th = self.theta_deg.to_radians();
                2.0 * self.a * (th.cos() + th.sin())
            }
        }
    }
    /// the four squares: centre, rotation sign
    pub fn squares(&self) -> Vec<([f64; 2], f64)> {
        let s = self.cell_side() / 2.0;
        vec![([-s / 2., -s / 2.], 1.), ([s / 2., s / 2.], 1.), ([-s / 2., s / 2.], -1.), ([s / 2., -s / 2.], -1.)]
    }
    /// corners of one rotated square
    pub fn square_corners(&self, c: [f64; 2], sign: f64) -> Vec<[f64; 2]> {
        let th = sign * self.theta_deg.to_radians();
        let h = self.a / 2.0;
        [[h, h], [-h, h], [-h, -h], [h, -h]]
            .iter()
            .map(|p| [c[0] + p[0] * th.cos() - p[1] * th.sin(), c[1] + p[0] * th.sin() + p[1] * th.cos()])
            .collect()
    }
    /// hinge centres (unique square corners)
    pub fn hinges(&self) -> Vec<[f64; 2]> {
        let mut out: Vec<[f64; 2]> = Vec::new();
        let tol = 1e-9 * self.a.max(1.0);
        for (c, s) in self.squares() {
            for p in self.square_corners(c, s) {
                if !out.iter().any(|q| (q[0] - p[0]).abs() < tol && (q[1] - p[1]).abs() < tol) {
                    out.push(p);
                }
            }
        }
        out
    }
    /// approximate solid fraction (squares + hinge disks, disks counted half inside)
    pub fn solid_fraction(&self) -> f64 {
        let l = self.cell_side();
        match self.kind {
            CellKind::Homogeneous => 1.0,
            // the hinge disks lie almost entirely inside the squares (Gmsh: 401.1 vs 400 for a = 10),
            // so the four squares give the solid area; the exact value comes back in geometry.json
            CellKind::RotatingSquares => (4.0 * self.a * self.a / (l * l)).min(1.0),
        }
    }
    /// feasibility constraints of a parameter set (empty = feasible)
    pub fn feasibility(&self, mesh: &MeshSettings) -> Vec<String> {
        let mut v = Vec::new();
        if !(self.a > 0.0) {
            v.push("square side a must be positive".into());
        }
        if self.kind == CellKind::RotatingSquares {
            if !(self.theta_deg >= 0.0 && self.theta_deg < 45.0) {
                v.push("rotation must be in [0, 45) degrees".into());
            }
            if !(self.t > 0.0) {
                v.push("hinge diameter t must be positive".into());
            }
            if self.t >= 0.5 * self.a {
                v.push("hinge diameter t must be below a/2 (hinges would merge with the squares)".into());
            }
            if mesh.size > self.t {
                v.push(format!("mesh size {} is larger than the hinge diameter {}: the hinge gets no element across", mesh.size, self.t));
            }
        }
        if !(mesh.size > 0.0) {
            v.push("mesh size must be positive".into());
        }
        if !(mesh.order == 1 || mesh.order == 2) {
            v.push("element order must be 1 or 2".into());
        }
        v
    }
}

impl Material {
    /// plane-stress stiffness in Voigt form (xx, yy, engineering xy)
    pub fn plane_stress_c(&self) -> [[f64; 3]; 3] {
        let f = self.e / (1.0 - self.nu * self.nu);
        [[f, f * self.nu, 0.], [f * self.nu, f, 0.], [0., 0., f * (1.0 - self.nu) / 2.0]]
    }
}

impl LoadCase {
    pub fn strain_cases(&self) -> Vec<[f64; 3]> {
        match *self {
            LoadCase::Homogenize { strain } => vec![[strain, 0., 0.], [0., strain, 0.], [0., 0., strain]],
            LoadCase::Strain { exx, eyy, gxy } => vec![[exx, eyy, gxy]],
        }
    }
}

impl Job {
    /// everything the engine scripts read (written as params.json into each bundle)
    pub fn params_json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": match self.cell.kind { CellKind::RotatingSquares => "rotating_squares", CellKind::Homogeneous => "homogeneous" },
            "a": self.cell.a,
            "theta_deg": self.cell.theta_deg,
            "t": self.cell.t,
            "L": self.cell.cell_side(),
            "E": self.material.e,
            "nu": self.material.nu,
            "thickness": self.material.thickness,
            "mesh_size": self.mesh.size,
            "order": self.mesh.order,
            "seed": self.mesh.seed,
            "has_corners": self.cell.kind == CellKind::Homogeneous,
            "strain_cases": self.load.strain_cases(),
            "groups": {"left": 1, "right": 2, "bottom": 3, "top": 4, "solid": 10, "anchor": 20, "corner_tr": 21},
        })
    }
    /// stable case id: hash of the engine-independent parameters
    pub fn case_id(&self) -> String {
        let s = serde_json::to_string(&self.params_json()).unwrap_or_default();
        crate::client::sha256_hex(s.as_bytes())[..12].to_string()
    }
}
