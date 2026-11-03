//! Element formulations (see MECHANICS.md for the rules and their sources).

use crate::model::*;

// ---------------------------------------------------------------- small dense helpers

pub fn inv_small(n: usize, a: &[f64]) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if m[r * n + c].abs() > m[p * n + c].abs() {
                p = r;
            }
        }
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                m.swap(p * n + k, c * n + k);
                inv.swap(p * n + k, c * n + k);
            }
        }
        let d = 1.0 / m[c * n + c];
        for k in 0..n {
            m[c * n + k] *= d;
            inv[c * n + k] *= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * n + c];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[c * n + k];
                        inv[r * n + k] -= f * inv[c * n + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = norm(a);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Gauss–Legendre points/weights on [-1,1] (Newton iteration; same values as the
/// original's tables in GAUSSI 0x1400671ac).
/// 16-point row of the original's table (GAUSSI 0x1400671ac), positive half, outermost first.
const GAUSS16_X: [f64; 8] = [
    0.9894009349916499, 0.9445750230732326, 0.8656312023878318, 0.7554044083550030,
    0.6178762444026438, 0.4580167776572274, 0.2816035507792589, 0.0950125098376374,
];
const GAUSS16_W: [f64; 8] = [
    0.027152459411754096, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
];

pub fn gauss(n: usize) -> Vec<(f64, f64)> {
    if n <= 1 {
        return vec![(0.0, 2.0)];
    }
    if n == 16 {
        let mut v: Vec<(f64, f64)> = (0..8).map(|i| (-GAUSS16_X[i], GAUSS16_W[i])).collect();
        v.extend((0..8).rev().map(|i| (GAUSS16_X[i], GAUSS16_W[i])));
        return v;
    }
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let mut x = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        out.push((x, w));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

// ---------------------------------------------------------------- shapes

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    Line2,
    Line3,
    Tri3,
    Tri6,
    Quad4,
    Quad8,
    Tet4,
    Tet10,
    Hex8,
    Hex20,
}

const HEX_C: [[f64; 3]; 8] = [
    [-1., -1., -1.],
    [1., -1., -1.],
    [1., 1., -1.],
    [-1., 1., -1.],
    [-1., -1., 1.],
    [1., -1., 1.],
    [1., 1., 1.],
    [-1., 1., 1.],
];
const HEX20_MID: [[f64; 3]; 12] = [
    [0., -1., -1.],
    [1., 0., -1.],
    [0., 1., -1.],
    [-1., 0., -1.],
    [0., -1., 1.],
    [1., 0., 1.],
    [0., 1., 1.],
    [-1., 0., 1.],
    [-1., -1., 0.],
    [1., -1., 0.],
    [1., 1., 0.],
    [-1., 1., 0.],
];
pub const HEX20_EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];
pub const TET10_EDGES: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];
pub const QUAD8_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
pub const TRI6_EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];
const QC: [[f64; 2]; 4] = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]];

impl Shape {
    pub fn corners(self) -> usize {
        match self {
            Shape::Line2 | Shape::Line3 => 2,
            Shape::Tri3 | Shape::Tri6 => 3,
            Shape::Quad4 | Shape::Quad8 | Shape::Tet4 | Shape::Tet10 => 4,
            Shape::Hex8 | Shape::Hex20 => 8,
        }
    }
    pub fn edges(self) -> &'static [[usize; 2]] {
        match self {
            Shape::Hex20 => &HEX20_EDGES,
            Shape::Tet10 => &TET10_EDGES,
            Shape::Quad8 => &QUAD8_EDGES,
            Shape::Tri6 => &TRI6_EDGES,
            _ => &[],
        }
    }

    /// shape functions and natural derivatives
    pub fn eval(self, p: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
        let (x, y, z) = (p[0], p[1], p[2]);
        match self {
            Shape::Line2 => (vec![0.5 * (1. - x), 0.5 * (1. + x)], vec![[-0.5, 0., 0.], [0.5, 0., 0.]]),
            Shape::Line3 => (
                vec![0.5 * x * (x - 1.), 0.5 * x * (x + 1.), 1. - x * x],
                vec![[x - 0.5, 0., 0.], [x + 0.5, 0., 0.], [-2. * x, 0., 0.]],
            ),
            Shape::Tri3 => (vec![1. - x - y, x, y], vec![[-1., -1., 0.], [1., 0., 0.], [0., 1., 0.]]),
            Shape::Tri6 => {
                let l = [1. - x - y, x, y];
                let dl = [[-1., -1.], [1., 0.], [0., 1.]];
                let mut n = Vec::with_capacity(6);
                let mut d = Vec::with_capacity(6);
                for i in 0..3 {
                    n.push(l[i] * (2. * l[i] - 1.));
                    let f = 4. * l[i] - 1.;
                    d.push([f * dl[i][0], f * dl[i][1], 0.]);
                }
                for e in TRI6_EDGES {
                    let (i, j) = (e[0], e[1]);
                    n.push(4. * l[i] * l[j]);
                    d.push([
                        4. * (dl[i][0] * l[j] + l[i] * dl[j][0]),
                        4. * (dl[i][1] * l[j] + l[i] * dl[j][1]),
                        0.,
                    ]);
                }
                (n, d)
            }
            Shape::Quad4 => {
                let mut n = Vec::with_capacity(4);
                let mut d = Vec::with_capacity(4);
                for c in QC {
                    n.push(0.25 * (1. + c[0] * x) * (1. + c[1] * y));
                    d.push([0.25 * c[0] * (1. + c[1] * y), 0.25 * c[1] * (1. + c[0] * x), 0.]);
                }
                (n, d)
            }
            Shape::Quad8 => {
                let mut n = Vec::with_capacity(8);
                let mut d = Vec::with_capacity(8);
                for c in QC {
                    let (a, b) = (c[0], c[1]);
                    n.push(0.25 * (1. + a * x) * (1. + b * y) * (a * x + b * y - 1.));
                    d.push([
                        0.25 * a * (1. + b * y) * (2. * a * x + b * y),
                        0.25 * b * (1. + a * x) * (a * x + 2. * b * y),
                        0.,
                    ]);
                }
                for c in [[0., -1.], [1., 0.], [0., 1.], [-1., 0.]] {
                    if c[0] == 0. {
                        let b = c[1];
                        n.push(0.5 * (1. - x * x) * (1. + b * y));
                        d.push([-x * (1. + b * y), 0.5 * b * (1. - x * x), 0.]);
                    } else {
                        let a = c[0];
                        n.push(0.5 * (1. + a * x) * (1. - y * y));
                        d.push([0.5 * a * (1. - y * y), -y * (1. + a * x), 0.]);
                    }
                }
                (n, d)
            }
            Shape::Tet4 => (
                vec![1. - x - y - z, x, y, z],
                vec![[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            ),
            Shape::Tet10 => {
                let l = [1. - x - y - z, x, y, z];
                let dl = [[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
                let mut n = Vec::with_capacity(10);
                let mut d = Vec::with_capacity(10);
                for i in 0..4 {
                    n.push(l[i] * (2. * l[i] - 1.));
                    let f = 4. * l[i] - 1.;
                    d.push([f * dl[i][0], f * dl[i][1], f * dl[i][2]]);
                }
                for e in TET10_EDGES {
                    let (i, j) = (e[0], e[1]);
                    n.push(4. * l[i] * l[j]);
                    let mut g = [0.; 3];
                    for k in 0..3 {
                        g[k] = 4. * (dl[i][k] * l[j] + l[i] * dl[j][k]);
                    }
                    d.push(g);
                }
                (n, d)
            }
            Shape::Hex8 => {
                let mut n = Vec::with_capacity(8);
                let mut d = Vec::with_capacity(8);
                for c in HEX_C {
                    let (a, b, cc) = (1. + c[0] * x, 1. + c[1] * y, 1. + c[2] * z);
                    n.push(0.125 * a * b * cc);
                    d.push([0.125 * c[0] * b * cc, 0.125 * a * c[1] * cc, 0.125 * a * b * c[2]]);
                }
                (n, d)
            }
            Shape::Hex20 => {
                let mut n = Vec::with_capacity(20);
                let mut d = Vec::with_capacity(20);
                for c in HEX_C {
                    let (a, b, cc) = (1. + c[0] * x, 1. + c[1] * y, 1. + c[2] * z);
                    let s = c[0] * x + c[1] * y + c[2] * z - 2.;
                    n.push(0.125 * a * b * cc * s);
                    d.push([
                        0.125 * c[0] * b * cc * (s + a),
                        0.125 * c[1] * a * cc * (s + b),
                        0.125 * c[2] * a * b * (s + cc),
                    ]);
                }
                for c in HEX20_MID {
                    if c[0] == 0. {
                        let (b, cc) = (1. + c[1] * y, 1. + c[2] * z);
                        n.push(0.25 * (1. - x * x) * b * cc);
                        d.push([-0.5 * x * b * cc, 0.25 * (1. - x * x) * c[1] * cc, 0.25 * (1. - x * x) * b * c[2]]);
                    } else if c[1] == 0. {
                        let (a, cc) = (1. + c[0] * x, 1. + c[2] * z);
                        n.push(0.25 * (1. - y * y) * a * cc);
                        d.push([0.25 * (1. - y * y) * c[0] * cc, -0.5 * y * a * cc, 0.25 * (1. - y * y) * a * c[2]]);
                    } else {
                        let (a, b) = (1. + c[0] * x, 1. + c[1] * y);
                        n.push(0.25 * (1. - z * z) * a * b);
                        d.push([0.25 * (1. - z * z) * c[0] * b, 0.25 * (1. - z * z) * a * c[1], -0.5 * z * a * b]);
                    }
                }
                (n, d)
            }
        }
    }
}

fn rule_line(n: usize) -> Vec<([f64; 3], f64)> {
    gauss(n).into_iter().map(|(x, w)| ([x, 0., 0.], w)).collect()
}
fn rule_quad(n: usize) -> Vec<([f64; 3], f64)> {
    let g = gauss(n);
    let mut r = Vec::new();
    for &(x, wx) in &g {
        for &(y, wy) in &g {
            r.push(([x, y, 0.], wx * wy));
        }
    }
    r
}
fn rule_hex(n: usize) -> Vec<([f64; 3], f64)> {
    let g = gauss(n);
    let mut r = Vec::new();
    for &(x, wx) in &g {
        for &(y, wy) in &g {
            for &(z, wz) in &g {
                r.push(([x, y, z], wx * wy * wz));
            }
        }
    }
    r
}
/// 14-point rule for the 20-node brick, exactly as stored in the original:
/// constants at 0x155acce48..0x155acce70 (b, a, wb, wa, 13 significant digits) and the point table
/// x at 0x157d3eca0, y at 0x157d3ed10, z at 0x157d3ed80: 8 corner points in brick node order
/// (I J K L M N O P), then -/+ ... along x, y and z (+a first).
const HEX14_A: f64 = 0.7958224257542;
const HEX14_B: f64 = 0.7587869106393;
const HEX14_WA: f64 = 0.8864265927978;
const HEX14_WB: f64 = 0.3351800554017;
fn rule_hex14() -> Vec<([f64; 3], f64)> {
    let (a, b) = (HEX14_A, HEX14_B);
    let mut r: Vec<([f64; 3], f64)> = HEX_C.iter().map(|c| ([c[0] * b, c[1] * b, c[2] * b], HEX14_WB)).collect();
    for k in 0..3 {
        for sgn in [1., -1.] {
            let mut p = [0.; 3];
            p[k] = sgn * a;
            r.push((p, HEX14_WA));
        }
    }
    r
}
const TRI_A: f64 = 2. / 3.;
const TRI_B: f64 = 1. / 6.;
/// 3-point triangle rule, point j has area coordinate L_j = 2/3
fn rule_tri3() -> Vec<([f64; 3], f64)> {
    vec![([TRI_B, TRI_B, 0.], 1. / 6.), ([TRI_A, TRI_B, 0.], 1. / 6.), ([TRI_B, TRI_A, 0.], 1. / 6.)]
}
pub const TET_A: f64 = 0.5854101966249685;
pub const TET_B: f64 = 0.1381966011250105;
/// 4-point tet rule, point j has volume coordinate L_j = a
fn rule_tet4() -> Vec<([f64; 3], f64)> {
    let (a, b) = (TET_A, TET_B);
    vec![([b, b, b], 1. / 24.), ([a, b, b], 1. / 24.), ([b, a, b], 1. / 24.), ([b, b, a], 1. / 24.)]
}

// ---------------------------------------------------------------- materials

#[derive(Clone, Copy, Debug)]
pub struct Iso {
    pub e: f64,
    pub nu: f64,
    pub rho: f64,
}

pub fn iso(m: &Model, mat: usize) -> Result<Iso, String> {
    let e = m.mat_prop(mat, "EX").ok_or_else(|| format!("Material {mat} has no EX (Young's modulus) defined."))?;
    // undefined PRXY/NUXY/GXY: GXY defaults to EX/2.6 (string 0x156321fa0), i.e. a Poisson's ratio of 0.3
    const GXY_DIVISOR: f64 = 2.6;
    let nu = m
        .mat_prop(mat, "PRXY")
        .or_else(|| m.mat_prop(mat, "NUXY"))
        .or_else(|| m.mat_prop(mat, "GXY").map(|g| e / (2.0 * g) - 1.0))
        .unwrap_or_else(|| e / (2.0 * (e / GXY_DIVISOR)) - 1.0);
    let rho = m.mat_prop(mat, "DENS").unwrap_or(0.0);
    Ok(Iso { e, nu, rho })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Plane {
    Solid3D,
    Stress,
    Strain,
    Axi,
}

fn dmat(mat: Iso, pl: Plane) -> Vec<f64> {
    let (e, nu) = (mat.e, mat.nu);
    let lam = e * nu / ((1. + nu) * (1. - 2. * nu));
    let mu = e / (2. * (1. + nu));
    match pl {
        Plane::Solid3D => {
            let mut d = vec![0.; 36];
            for i in 0..3 {
                for j in 0..3 {
                    d[i * 6 + j] = lam;
                }
                d[i * 6 + i] = lam + 2. * mu;
                d[(i + 3) * 6 + i + 3] = mu;
            }
            d
        }
        Plane::Strain | Plane::Axi => {
            let mut d = vec![0.; 16];
            for i in 0..3 {
                for j in 0..3 {
                    d[i * 4 + j] = lam;
                }
                d[i * 4 + i] = lam + 2. * mu;
            }
            d[15] = mu;
            d
        }
        Plane::Stress => {
            let c = e / (1. - nu * nu);
            let mut d = vec![0.; 16];
            d[0] = c;
            d[1] = c * nu;
            d[4] = c * nu;
            d[5] = c;
            d[15] = mu;
            d
        }
    }
}

// ---------------------------------------------------------------- continuum elements

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tech {
    Std,
    BBar,
    Enhanced,
}

struct Cont {
    shape: Shape,
    ids: Vec<usize>,
    x: Vec<[f64; 3]>,
    plane: Plane,
    thick: f64,
    tech: Tech,
    d: Vec<f64>,
    rule: Vec<([f64; 3], f64)>,
    rec: Vec<[f64; 3]>,
    mat: Iso,
    enh_vol: bool,
    eresx_copy: bool,
}

pub struct ElemMat {
    pub dofs: Vec<(usize, u8)>,
    pub k: Vec<f64>,
    pub f: Vec<f64>,
}

#[derive(Default, Clone)]
pub struct ElemOut {
    /// node id -> stress SX SY SZ SXY SYZ SXZ (corner-extrapolated, averaged within element)
    pub nodal: Vec<(usize, [f64; 6])>,
    /// LINK: [axial force, axial stress]; BEAM: local end forces I(6) J(6); SPRING: [force]
    pub forces: Vec<f64>,
}

/// cell shape, node ids (shape order), plane mode and thickness of a continuum element (for exporters)
pub fn continuum_cell(m: &Model, e: &Element) -> Option<(Shape, Vec<usize>, Plane, f64)> {
    let k = m.kind_of(e)?;
    if !(k.is_solid() || k.is_plane()) {
        return None;
    }
    let c = prepare(m, e, k).ok()?;
    let mut shape = c.shape;
    let mut ids = c.ids;
    // a 4-node quad with K = L is a triangle
    if shape == Shape::Quad4 && ids[2] == ids[3] {
        shape = Shape::Tri3;
        ids.truncate(3);
    }
    Some((shape, ids, c.plane, c.thick))
}

fn same(a: &[usize]) -> bool {
    a.iter().all(|&x| x == a[0])
}

fn prepare(m: &Model, e: &Element, kind: Kind) -> Result<Cont, String> {
    let n = &e.nodes;
    let k = |i: usize| m.keyopt(e, i);
    let mat = iso(m, e.mat)?;
    let shape;
    let ids: Vec<usize>;
    let plane;
    let tech;
    let rule;
    let rec: Vec<[f64; 3]>;
    let thick;
    let g3 = 1. / 3f64.sqrt();
    let hex_rec: Vec<[f64; 3]> = HEX_C.iter().map(|c| [c[0] * g3, c[1] * g3, c[2] * g3]).collect();
    let quad_rec: Vec<[f64; 3]> = QC.iter().map(|c| [c[0] * g3, c[1] * g3, 0.]).collect();
    match kind {
        Kind::Solid185 => {
            if n.len() < 8 {
                return Err("SOLID185 needs 8 nodes".into());
            }
            if n[2] == n[3] && same(&n[4..8]) {
                shape = Shape::Tet4;
                ids = vec![n[0], n[1], n[2], n[4]];
                rule = vec![([0.25, 0.25, 0.25], 1. / 6.)];
                rec = vec![[0.25, 0.25, 0.25]];
                tech = Tech::Std;
            } else {
                shape = Shape::Hex8;
                ids = n[..8].to_vec();
                rule = rule_hex(2);
                rec = hex_rec;
                let degenerate = n[2] == n[3] || n[6] == n[7] || same(&n[4..8]);
                tech = match k(2) {
                    2 | 3 if !degenerate => Tech::Enhanced,
                    _ => Tech::BBar,
                };
            }
            plane = Plane::Solid3D;
            thick = 1.;
        }
        Kind::Solid186 => {
            if n.len() < 20 || n.iter().take(20).any(|&x| x == 0) {
                return Err("SOLID186 needs 20 nodes".into());
            }
            plane = Plane::Solid3D;
            thick = 1.;
            tech = Tech::Std;
            if n[2] == n[3] && same(&n[4..8]) {
                shape = Shape::Tet10;
                ids = vec![n[0], n[1], n[2], n[4], n[8], n[9], n[11], n[16], n[17], n[18]];
                rule = rule_tet4();
                rec = rule.iter().map(|r| r.0).collect();
            } else {
                shape = Shape::Hex20;
                ids = n[..20].to_vec();
                let degenerate = n[2] == n[3] || n[6] == n[7] || same(&n[4..8]);
                rule = if degenerate { rule_hex(3) } else if k(2) == 1 { rule_hex14() } else { rule_hex(2) };
                // full integration: stresses extrapolated from the 8 corner points of the 14-point rule
                rec = if k(2) == 1 && !degenerate {
                    HEX_C.iter().map(|c| [c[0] * HEX14_B, c[1] * HEX14_B, c[2] * HEX14_B]).collect()
                } else {
                    hex_rec
                };
            }
        }
        Kind::Solid187 => {
            if n.len() < 10 || n.iter().take(10).any(|&x| x == 0) {
                return Err("SOLID187 needs 10 nodes".into());
            }
            shape = Shape::Tet10;
            ids = n[..10].to_vec();
            rule = rule_tet4();
            rec = rule.iter().map(|r| r.0).collect();
            plane = Plane::Solid3D;
            thick = 1.;
            tech = Tech::Std;
        }
        Kind::Plane182 | Kind::Plane183 => {
            plane = match k(3) {
                1 => Plane::Axi,
                2 | 5 => Plane::Strain,
                _ => Plane::Stress,
            };
            thick = if k(3) == 3 {
                m.reals.get(&e.real).and_then(|r| r.first().copied()).filter(|t| *t > 0.).unwrap_or(1.)
            } else {
                1.
            };
            if kind == Kind::Plane182 {
                if n.len() < 4 {
                    return Err("PLANE182 needs 4 nodes".into());
                }
                shape = Shape::Quad4;
                ids = n[..4].to_vec();
                rule = rule_quad(2);
                rec = quad_rec;
                let tri = n[2] == n[3];
                tech = match k(1) {
                    2 | 3 if !tri => Tech::Enhanced,
                    _ if plane == Plane::Stress => Tech::Std,
                    _ => Tech::BBar,
                };
            } else {
                tech = Tech::Std;
                let degen_tri = n.len() >= 8 && n[2] == n[3] && n[3] == n[6];
                let tri6 = degen_tri || (n.len() >= 6 && (n.len() < 8 || n[6] == 0));
                if tri6 {
                    shape = Shape::Tri6;
                    ids = if degen_tri { vec![n[0], n[1], n[2], n[4], n[5], n[7]] } else { n[..6].to_vec() };
                    rule = rule_tri3();
                    rec = rule.iter().map(|r| r.0).collect();
                } else {
                    if n.len() < 8 || n.iter().take(8).any(|&x| x == 0) {
                        return Err("PLANE183 needs 8 nodes".into());
                    }
                    shape = Shape::Quad8;
                    ids = n[..8].to_vec();
                    rule = rule_quad(2);
                    rec = quad_rec;
                }
            }
        }
        _ => return Err("not a continuum element".into()),
    }
    let mut x = Vec::with_capacity(ids.len());
    for &id in &ids {
        x.push(m.node(id).ok_or_else(|| format!("node {id} is not defined"))?);
    }
    let enh_vol = kind == Kind::Solid185 && tech == Tech::Enhanced && k(2) == 2;
    Ok(Cont { shape, ids, x, plane, thick, tech, d: dmat(mat, plane), rule, rec, mat, enh_vol, eresx_copy: m.eresx_copy })
}

struct GP {
    n: Vec<f64>,
    dndx: Vec<[f64; 3]>,
    detj: f64,
    r: f64,
}

type Cond = Option<(Vec<f64>, Vec<f64>)>;

impl Cont {
    fn ndim(&self) -> usize {
        if self.plane == Plane::Solid3D {
            3
        } else {
            2
        }
    }
    fn ncomp(&self) -> usize {
        if self.plane == Plane::Solid3D {
            6
        } else {
            4
        }
    }
    fn jac(&self, dn: &[[f64; 3]]) -> (Vec<f64>, f64) {
        let nd = self.ndim();
        let mut j = vec![0.; nd * nd];
        for (a, g) in dn.iter().enumerate() {
            for i in 0..nd {
                for k in 0..nd {
                    j[i * nd + k] += g[i] * self.x[a][k];
                }
            }
        }
        let det = if nd == 3 {
            j[0] * (j[4] * j[8] - j[5] * j[7]) - j[1] * (j[3] * j[8] - j[5] * j[6]) + j[2] * (j[3] * j[7] - j[4] * j[6])
        } else {
            j[0] * j[3] - j[1] * j[2]
        };
        (j, det)
    }
    fn gp(&self, p: [f64; 3]) -> Result<GP, String> {
        let (n, dn) = self.shape.eval(p);
        let nd = self.ndim();
        let (j, det) = self.jac(&dn);
        if !(det > 0.0) {
            return Err(format!(
                "negative or zero Jacobian (element shape is inverted, collapsed or nodes are ordered clockwise), det = {det:.3e}"
            ));
        }
        let ji = inv_small(nd, &j).ok_or("singular Jacobian")?;
        let mut dndx = Vec::with_capacity(n.len());
        for g in &dn {
            let mut o = [0.; 3];
            for k in 0..nd {
                for i in 0..nd {
                    o[k] += ji[k * nd + i] * g[i];
                }
            }
            dndx.push(o);
        }
        let r = if self.plane == Plane::Axi { n.iter().zip(&self.x).map(|(a, x)| a * x[0]).sum() } else { 0. };
        Ok(GP { n, dndx, detj: det, r })
    }
    fn wfac(&self, g: &GP, w: f64) -> f64 {
        match self.plane {
            Plane::Solid3D => g.detj * w,
            Plane::Axi => g.detj * w * 2. * std::f64::consts::PI * g.r,
            _ => g.detj * w * self.thick,
        }
    }
    /// strain-displacement matrix (ncomp x ndim*nn), row major
    fn bmat(&self, n: &[f64], dndx: &[[f64; 3]], r: f64) -> Vec<f64> {
        let nd = self.ndim();
        let nc = self.ncomp();
        let cols = nd * n.len();
        let mut b = vec![0.; nc * cols];
        for a in 0..n.len() {
            let g = dndx[a];
            let c = a * nd;
            if nd == 3 {
                b[c] = g[0];
                b[cols + c + 1] = g[1];
                b[2 * cols + c + 2] = g[2];
                b[3 * cols + c] = g[1];
                b[3 * cols + c + 1] = g[0];
                b[4 * cols + c + 1] = g[2];
                b[4 * cols + c + 2] = g[1];
                b[5 * cols + c] = g[2];
                b[5 * cols + c + 2] = g[0];
            } else {
                b[c] = g[0];
                b[cols + c + 1] = g[1];
                if self.plane == Plane::Axi && r > 0. {
                    b[2 * cols + c] = n[a] / r;
                }
                b[3 * cols + c] = g[1];
                b[3 * cols + c + 1] = g[0];
            }
        }
        b
    }
    fn volvec(&self, g: &GP) -> Vec<f64> {
        let nd = self.ndim();
        let mut v = vec![0.; nd * g.n.len()];
        for a in 0..g.n.len() {
            for k in 0..nd {
                v[a * nd + k] = g.dndx[a][k];
            }
            if self.plane == Plane::Axi && g.r > 0. {
                v[a * nd] += g.n[a] / g.r;
            }
        }
        v
    }
    fn bbar_avg(&self) -> Result<Vec<f64>, String> {
        let mut acc = vec![0.; self.ndim() * self.ids.len()];
        let mut vol = 0.;
        for &(p, w) in &self.rule {
            let g = self.gp(p)?;
            let f = self.wfac(&g, w);
            let v = self.volvec(&g);
            for (a, b) in acc.iter_mut().zip(&v) {
                *a += b * f;
            }
            vol += f;
        }
        for a in acc.iter_mut() {
            *a /= vol;
        }
        Ok(acc)
    }
    fn apply_bbar(&self, b: &mut [f64], g: &GP, avg: &[f64]) {
        let cols = avg.len();
        let v = self.volvec(g);
        // plane strain: dilatation averaged in-plane only (matches the original to 5 digits)
        let two = self.plane == Plane::Strain;
        for c in 0..cols {
            let dv = (avg[c] - v[c]) / if two { 2. } else { 3. };
            for r in 0..(if two { 2 } else { 3 }) {
                b[r * cols + c] += dv;
            }
        }
    }
    fn n_enh(&self) -> usize {
        let nd = self.ndim();
        nd * nd + if self.enh_vol { 4 } else { 0 }
    }
    /// incompatible-mode matrix G (ncomp x nmodes*ndim) at point p
    fn gmat(&self, p: [f64; 3], g: &GP) -> Result<Vec<f64>, String> {
        let nd = self.ndim();
        let (_, dn0) = self.shape.eval([0., 0., 0.]);
        let (j0, det0) = self.jac(&dn0);
        let j0i = inv_small(nd, &j0).ok_or("singular Jacobian")?;
        let ratio = det0 / g.detj;
        let mut pn = Vec::with_capacity(nd);
        let mut pd = Vec::with_capacity(nd);
        for k in 0..nd {
            pn.push(1. - p[k] * p[k]);
            let mut d = [0.; 3];
            d[k] = -2. * p[k];
            let mut o = [0.; 3];
            for a in 0..nd {
                for i in 0..nd {
                    o[a] += j0i[a * nd + i] * d[i];
                }
                o[a] *= ratio;
            }
            pd.push(o);
        }
        let mut gm = self.bmat(&pn, &pd, g.r);
        if self.enh_vol {
            // SOLID185 KEYOPT(2)=2: 4 extra volumetric modes (13 internal DOFs in total)
            let (x, y, z) = (p[0], p[1], p[2]);
            let fs = [x * y, y * z, z * x, x * y * z];
            let nc = self.ncomp();
            let old = gm.len() / nc;
            let new = old + 4;
            let mut g2 = vec![0.; nc * new];
            for r in 0..nc {
                for c in 0..old {
                    g2[r * new + c] = gm[r * old + c];
                }
            }
            for (k, f) in fs.iter().enumerate() {
                for r in 0..3 {
                    g2[r * new + old + k] = ratio * f;
                }
            }
            gm = g2;
        }
        Ok(gm)
    }

    fn stiffness(&self, acel: [f64; 3]) -> Result<(Vec<f64>, Vec<f64>, Cond), String> {
        let nd = self.ndim();
        let nc = self.ncomp();
        let ndof = nd * self.ids.len();
        let mut k = vec![0.; ndof * ndof];
        let mut f = vec![0.; ndof];
        let avg = if self.tech == Tech::BBar { Some(self.bbar_avg()?) } else { None };
        let na = if self.tech == Tech::Enhanced { self.n_enh() } else { 0 };
        let mut kaa = vec![0.; na * na];
        let mut kua = vec![0.; ndof * na];
        let body = [-acel[0] * self.mat.rho, -acel[1] * self.mat.rho, -acel[2] * self.mat.rho];
        let has_body = body.iter().any(|v| *v != 0.);
        let mut db = vec![0.; nc * ndof];
        for &(p, w) in &self.rule {
            let g = self.gp(p)?;
            let fw = self.wfac(&g, w);
            let mut b = self.bmat(&g.n, &g.dndx, g.r);
            if let Some(avg) = &avg {
                self.apply_bbar(&mut b, &g, avg);
            }
            for r in 0..nc {
                for c in 0..ndof {
                    let mut s = 0.;
                    for q in 0..nc {
                        s += self.d[r * nc + q] * b[q * ndof + c];
                    }
                    db[r * ndof + c] = s * fw;
                }
            }
            for i in 0..ndof {
                for j in i..ndof {
                    let mut s = 0.;
                    for q in 0..nc {
                        s += b[q * ndof + i] * db[q * ndof + j];
                    }
                    k[i * ndof + j] += s;
                }
            }
            if na > 0 {
                let gm = self.gmat(p, &g)?;
                for i in 0..ndof {
                    for j in 0..na {
                        let mut s = 0.;
                        for q in 0..nc {
                            s += db[q * ndof + i] * gm[q * na + j];
                        }
                        kua[i * na + j] += s;
                    }
                }
                for i in 0..na {
                    for j in 0..na {
                        let mut s = 0.;
                        for q in 0..nc {
                            let mut dg = 0.;
                            for t in 0..nc {
                                dg += self.d[q * nc + t] * gm[t * na + j];
                            }
                            s += gm[q * na + i] * dg;
                        }
                        kaa[i * na + j] += s * fw;
                    }
                }
            }
            if has_body {
                for a in 0..self.ids.len() {
                    for c in 0..nd {
                        f[a * nd + c] += g.n[a] * body[c] * fw;
                    }
                }
            }
        }
        for i in 0..ndof {
            for j in 0..i {
                k[i * ndof + j] = k[j * ndof + i];
            }
        }
        let mut cond = None;
        if na > 0 {
            let kai = inv_small(na, &kaa).ok_or("singular enhanced-strain matrix")?;
            let mut x = vec![0.; na * ndof];
            for i in 0..na {
                for j in 0..ndof {
                    let mut s = 0.;
                    for q in 0..na {
                        s += kai[i * na + q] * kua[j * na + q];
                    }
                    x[i * ndof + j] = s;
                }
            }
            for i in 0..ndof {
                for j in 0..ndof {
                    let mut s = 0.;
                    for q in 0..na {
                        s += kua[i * na + q] * x[q * ndof + j];
                    }
                    k[i * ndof + j] -= s;
                }
            }
            cond = Some((x, kua));
        }
        Ok((k, f, cond))
    }

    fn stresses(&self, u: &[f64]) -> Result<Vec<[f64; 6]>, String> {
        let nd = self.ndim();
        let nc = self.ncomp();
        let ndof = nd * self.ids.len();
        let avg = if self.tech == Tech::BBar { Some(self.bbar_avg()?) } else { None };
        let alpha = if self.tech == Tech::Enhanced {
            let (_, _, cond) = self.stiffness([0.; 3])?;
            let (x, _) = cond.unwrap();
            let na = self.n_enh();
            let mut a = vec![0.; na];
            for i in 0..na {
                a[i] = -(0..ndof).map(|j| x[i * ndof + j] * u[j]).sum::<f64>();
            }
            Some(a)
        } else {
            None
        };
        let mut out = Vec::with_capacity(self.rec.len());
        for &p in &self.rec {
            let g = self.gp(p)?;
            let mut b = self.bmat(&g.n, &g.dndx, g.r);
            if let Some(avg) = &avg {
                self.apply_bbar(&mut b, &g, avg);
            }
            let mut eps = vec![0.; nc];
            for r in 0..nc {
                eps[r] = (0..ndof).map(|c| b[r * ndof + c] * u[c]).sum();
            }
            if let Some(a) = &alpha {
                let gm = self.gmat(p, &g)?;
                let na = a.len();
                for r in 0..nc {
                    eps[r] += (0..na).map(|c| gm[r * na + c] * a[c]).sum::<f64>();
                }
            }
            let mut s = [0.; 6];
            for r in 0..nc {
                s[r] = (0..nc).map(|c| self.d[r * nc + c] * eps[c]).sum();
            }
            out.push(s);
        }
        // extrapolate to corners (rule point j <-> corner j for tets/triangles)
        let nco = self.shape.corners();
        let mut corner = vec![[0.; 6]; nco];
        let s3 = 3f64.sqrt();
        let copy = self.eresx_copy && out.len() >= nco;
        if copy {
            // ERESX,NO: each corner takes the value of its own (nearest) integration point
            for c in 0..nco {
                corner[c] = out[c];
            }
        }
        match if copy { Shape::Line2 } else { self.shape } {
            Shape::Hex8 | Shape::Hex20 => {
                let sc = 1. / self.rec[6][0].abs();
                for (c, cc) in HEX_C.iter().enumerate() {
                    let (w, _) = Shape::Hex8.eval([cc[0] * sc, cc[1] * sc, cc[2] * sc]);
                    for (gpi, wi) in w.iter().enumerate() {
                        for q in 0..6 {
                            corner[c][q] += wi * out[gpi][q];
                        }
                    }
                }
            }
            Shape::Quad4 | Shape::Quad8 => {
                for (c, cc) in QC.iter().enumerate() {
                    let (w, _) = Shape::Quad4.eval([cc[0] * s3, cc[1] * s3, 0.]);
                    for (gpi, wi) in w.iter().enumerate() {
                        for q in 0..6 {
                            corner[c][q] += wi * out[gpi][q];
                        }
                    }
                }
            }
            Shape::Tet10 | Shape::Tri6 => {
                let (a, b) = if self.shape == Shape::Tet10 { (TET_A, TET_B) } else { (TRI_A, TRI_B) };
                for c in 0..nco {
                    for j in 0..nco {
                        let wi = ((if c == j { 1. } else { 0. }) - b) / (a - b);
                        for q in 0..6 {
                            corner[c][q] += wi * out[j][q];
                        }
                    }
                }
            }
            _ if copy => {}
            _ => {
                for c in 0..nco {
                    corner[c] = out[0];
                }
            }
        }
        let mut all = corner.clone();
        for e in self.shape.edges() {
            let mut s = [0.; 6];
            for q in 0..6 {
                s[q] = 0.5 * (corner[e[0]][q] + corner[e[1]][q]);
            }
            all.push(s);
        }
        Ok(all)
    }
}

// ---------------------------------------------------------------- beams / links

/// residual-flexibility constant of linear BEAM188 (fitted to the original's results)
const BEAM_RBF: f64 = 1. / 96.;
/// shear correction the original reports for any rectangle ("Shear Correction-xy = 0.84211")
const RECT_SHEAR_CORR: f64 = 0.8421052631578947;
/// shear correction the original reports for a solid circle at nu = 0.3
const CSOLID_SHEAR_CORR: f64 = 0.85691;

/// area and second moment of a circle approximated by n quadratic boundary cells (mid-node on the circle)
fn curved_ngon(r: f64, n: usize) -> (f64, f64) {
    let m = 64; // points per cell edge on the parabola
    let mut pts = Vec::with_capacity(n * m);
    let pi = std::f64::consts::PI;
    for c in 0..n {
        let t0 = 2. * pi * c as f64 / n as f64;
        let t1 = 2. * pi * (c + 1) as f64 / n as f64;
        let p0 = [r * t0.cos(), r * t0.sin()];
        let p2 = [r * t1.cos(), r * t1.sin()];
        let tm = 0.5 * (t0 + t1);
        let p1 = [r * tm.cos(), r * tm.sin()];
        for k in 0..m {
            let x = -1. + 2. * k as f64 / m as f64;
            let n0 = 0.5 * x * (x - 1.);
            let n1 = 1. - x * x;
            let n2 = 0.5 * x * (x + 1.);
            pts.push([n0 * p0[0] + n1 * p1[0] + n2 * p2[0], n0 * p0[1] + n1 * p1[1] + n2 * p2[1]]);
        }
    }
    let (mut a, mut iy) = (0., 0.);
    for k in 0..pts.len() {
        let (p, q) = (pts[k], pts[(k + 1) % pts.len()]);
        let c = p[0] * q[1] - q[0] * p[1];
        a += c / 2.;
        iy += c * (p[1] * p[1] + p[1] * q[1] + q[1] * q[1]) / 12.;
    }
    (a, iy)
}

/// torsion constant of a rectangle as the original computes it: warping function on a
/// 2 x 2 mesh of 9-node cells (reproduces its 74629 for 20 x 40 and 115530 for 30 x 30)
fn rect_torsion(b: f64, h: f64) -> f64 {
    let (gx, gy) = (5usize, 5usize);
    let n = gx * gy;
    let pt = |i: usize, j: usize| [-b / 2. + b * i as f64 / 4., -h / 2. + h * j as f64 / 4.];
    let mut k = vec![0.; n * n];
    let mut f = vec![0.; n];
    let mut j0 = 0.;
    let g = gauss(3);
    let l = |t: f64| [t * (t - 1.) / 2., 1. - t * t, t * (t + 1.) / 2.];
    let dl = |t: f64| [t - 0.5, -2. * t, t + 0.5];
    for ej in 0..2 {
        for ei in 0..2 {
            let mut ids = Vec::with_capacity(9);
            let mut xe = Vec::with_capacity(9);
            for bj in 0..3 {
                for bi in 0..3 {
                    ids.push((2 * ei + bi) + gx * (2 * ej + bj));
                    xe.push(pt(2 * ei + bi, 2 * ej + bj));
                }
            }
            for &(xi, wx) in &g {
                for &(et, wy) in &g {
                    let (lx, ly, dx, dy) = (l(xi), l(et), dl(xi), dl(et));
                    let mut nn = [0.; 9];
                    let mut dn = [[0.; 2]; 9];
                    for bj in 0..3 {
                        for bi in 0..3 {
                            nn[bj * 3 + bi] = lx[bi] * ly[bj];
                            dn[bj * 3 + bi] = [dx[bi] * ly[bj], lx[bi] * dy[bj]];
                        }
                    }
                    let mut jm = [0.; 4];
                    let (mut y, mut z) = (0., 0.);
                    for q in 0..9 {
                        jm[0] += dn[q][0] * xe[q][0];
                        jm[1] += dn[q][0] * xe[q][1];
                        jm[2] += dn[q][1] * xe[q][0];
                        jm[3] += dn[q][1] * xe[q][1];
                        y += nn[q] * xe[q][0];
                        z += nn[q] * xe[q][1];
                    }
                    let det = jm[0] * jm[3] - jm[1] * jm[2];
                    let wt = det * wx * wy;
                    let mut dxy = [[0.; 2]; 9];
                    for q in 0..9 {
                        dxy[q] = [(jm[3] * dn[q][0] - jm[1] * dn[q][1]) / det, (-jm[2] * dn[q][0] + jm[0] * dn[q][1]) / det];
                    }
                    for a in 0..9 {
                        for c in 0..9 {
                            k[ids[a] * n + ids[c]] += wt * (dxy[a][0] * dxy[c][0] + dxy[a][1] * dxy[c][1]);
                        }
                        f[ids[a]] += wt * (dxy[a][0] * z - dxy[a][1] * y);
                    }
                    j0 += wt * (y * y + z * z);
                }
            }
        }
    }
    // pin node 0 (warping is defined up to a constant)
    for q in 0..n {
        k[q] = 0.;
        k[q * n] = 0.;
    }
    k[0] = 1.;
    f[0] = 0.;
    match inv_small(n, &k) {
        Some(ki) => {
            let psi: Vec<f64> = (0..n).map(|i| (0..n).map(|j| ki[i * n + j] * f[j]).sum()).collect();
            j0 - (0..n).map(|i| psi[i] * f[i]).sum::<f64>()
        }
        None => {
            let (aa, bb) = if b >= h { (b, h) } else { (h, b) };
            let r = bb / aa;
            aa * bb.powi(3) * (1. / 3. - 0.21 * r * (1. - r.powi(4) / 12.))
        }
    }
}

pub struct BeamSec {
    pub a: f64,
    pub iy: f64,
    pub iz: f64,
    pub j: f64,
    pub kappa: f64,
}

fn beam_section(m: &Model, e: &Element, nu: f64) -> Result<BeamSec, String> {
    if let Some(s) = m.sections.get(&e.sec).filter(|s| s.kind == "BEAM") {
        let d = |i: usize| s.data.get(i).copied().unwrap_or(0.);
        let pi = std::f64::consts::PI;
        return Ok(match s.subtype.as_str() {
            "RECT" | "" => {
                let (b, h) = (d(0), d(1));
                BeamSec {
                    a: b * h,
                    iz: h * b.powi(3) / 12.,
                    iy: b * h.powi(3) / 12.,
                    j: rect_torsion(b, h),
                    kappa: RECT_SHEAR_CORR,
                }
            }
            "CSOLID" => {
                // the original meshes the circle with 8 curved (quadratic) cells around
                let r = d(0);
                let ncell = if d(1) >= 4. { d(1) as usize } else { 8 };
                let (a, i) = curved_ngon(r, ncell);
                let _ = pi;
                BeamSec { a, iy: i, iz: i, j: 2. * i, kappa: if (nu - 0.3).abs() < 1e-9 { CSOLID_SHEAR_CORR } else { 6. * (1. + nu) / (7. + 6. * nu) * CSOLID_SHEAR_CORR / (7.8 / 8.8) } }
            }
            "CTUBE" => {
                let (ri, ro) = (d(0), d(1));
                let i = pi * (ro.powi(4) - ri.powi(4)) / 4.;
                let mm = ri / ro;
                let q = (1. + mm * mm).powi(2);
                BeamSec {
                    a: pi * (ro * ro - ri * ri),
                    iy: i,
                    iz: i,
                    j: 2. * i,
                    kappa: 6. * (1. + nu) * q / ((7. + 6. * nu) * q + (20. + 12. * nu) * mm * mm),
                }
            }
            "I" => {
                let (w1, w2, w3, t1, t2, t3) = (d(0), d(1), d(2), d(3), d(4), d(5));
                let hw = w3 - t1 - t2;
                let a = w1 * t1 + w2 * t2 + hw * t3;
                let zc = (w1 * t1 * t1 / 2. + hw * t3 * (t1 + hw / 2.) + w2 * t2 * (w3 - t2 / 2.)) / a;
                let iy = w1 * t1.powi(3) / 12.
                    + w1 * t1 * (t1 / 2. - zc).powi(2)
                    + t3 * hw.powi(3) / 12.
                    + hw * t3 * (t1 + hw / 2. - zc).powi(2)
                    + w2 * t2.powi(3) / 12.
                    + w2 * t2 * (w3 - t2 / 2. - zc).powi(2);
                let iz = t1 * w1.powi(3) / 12. + t2 * w2.powi(3) / 12. + hw * t3.powi(3) / 12.;
                let j = (w1 * t1.powi(3) + w2 * t2.powi(3) + hw * t3.powi(3)) / 3.;
                BeamSec { a, iy, iz, j, kappa: (hw * t3 / a).max(0.1) }
            }
            "HREC" => {
                let (w1, w2, t1, t2, t3, t4) = (d(0), d(1), d(2), d(3), d(4), d(5));
                let (bi, hi) = (w1 - t1 - t2, w2 - t3 - t4);
                let a = w1 * w2 - bi * hi;
                let iy = (w1 * w2.powi(3) - bi * hi.powi(3)) / 12.;
                let iz = (w2 * w1.powi(3) - hi * bi.powi(3)) / 12.;
                let tm = (t1 + t2 + t3 + t4) / 4.;
                let (bm, hm) = (w1 - (t1 + t2) / 2., w2 - (t3 + t4) / 2.);
                let j = 2. * tm * (bm * hm).powi(2) / (bm + hm);
                BeamSec { a, iy, iz, j, kappa: 0.5 }
            }
            "ASEC" => BeamSec { a: d(0), iy: d(1), iz: d(3), j: d(5), kappa: 5. / 6. },
            other => return Err(format!("beam section subtype {other} is not supported yet")),
        });
    }
    if let Some(r) = m.reals.get(&e.real) {
        let g = |i: usize| r.get(i).copied().unwrap_or(0.);
        let (a, iz, iy) = (g(0), g(1), g(2));
        let iy = if iy > 0. { iy } else { iz };
        let j = if g(6) > 0. { g(6) } else { iy + iz };
        return Ok(BeamSec { a, iy, iz, j, kappa: 0. });
    }
    Err(format!("beam element has no section (SECTYPE/SECDATA) for section {}", e.sec))
}

fn beam_frame(m: &Model, e: &Element, xi: [f64; 3], xj: [f64; 3]) -> Result<([[f64; 3]; 3], f64), String> {
    let d = sub(xj, xi);
    let l = norm(d);
    if l <= 0. {
        return Err("beam element has zero length".into());
    }
    let ex = unit(d);
    let (ey, ez);
    if let Some(xk) = e.nodes.get(2).filter(|&&k| k > 0).and_then(|&k| m.node(k)) {
        // node K lies in the element x-z plane
        let v = sub(xk, xi);
        let y = cross(v, ex);
        if norm(y) < 1e-12 * l {
            return Err("beam orientation node K is on the beam axis".into());
        }
        ey = unit(y);
        ez = cross(ex, ey);
    } else {
        let y = cross([0., 0., 1.], ex);
        ey = if norm(y) < 1e-8 { [0., 1., 0.] } else { unit(y) };
        ez = cross(ex, ey);
    }
    Ok(([ex, ey, ez], l))
}

fn transform(kl: &[f64], r: &[[f64; 3]; 3], n: usize) -> Vec<f64> {
    let nb = n / 3;
    let mut t = vec![0.; n * n];
    for b in 0..nb {
        for i in 0..3 {
            for j in 0..3 {
                t[(b * 3 + i) * n + b * 3 + j] = r[i][j];
            }
        }
    }
    let mut tmp = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.;
            for q in 0..n {
                s += kl[i * n + q] * t[q * n + j];
            }
            tmp[i * n + j] = s;
        }
    }
    let mut k = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.;
            for q in 0..n {
                s += t[q * n + i] * tmp[q * n + j];
            }
            k[i * n + j] = s;
        }
    }
    k
}

fn beam_local(m: &Model, e: &Element, l: f64) -> Result<Vec<f64>, String> {
    let mat = iso(m, e.mat)?;
    let s = beam_section(m, e, mat.nu)?;
    let g = mat.e / (2. * (1. + mat.nu));
    let mut k = vec![0.; 144];
    {
        let mut add = |dofs: &[usize], b: &[f64], c: f64| {
            for (i, &di) in dofs.iter().enumerate() {
                for (j, &dj) in dofs.iter().enumerate() {
                    k[di * 12 + dj] += c * b[i] * b[j];
                }
            }
        };
        add(&[0, 6], &[-1. / l, 1. / l], mat.e * s.a * l);
        add(&[3, 9], &[-1. / l, 1. / l], g * s.j * l);
        if s.kappa > 0. {
            // Timoshenko, linear, one integration point (BEAM188 KEYOPT(3)=0), plus the original's
            // extra per-element shear flexibility (1+nu)/(96 kappa) * L^2/(E I) (fitted to 6 runs, exact to 7 digits)
            let ks = |i: f64| 1. / (1. / (s.kappa * g * s.a) + BEAM_RBF * (1. + mat.nu) / s.kappa * l * l / (mat.e * i));
            add(&[5, 11], &[-1. / l, 1. / l], mat.e * s.iz * l);
            add(&[1, 5, 7, 11], &[-1. / l, -0.5, 1. / l, -0.5], ks(s.iz) * l);
            add(&[4, 10], &[-1. / l, 1. / l], mat.e * s.iy * l);
            add(&[2, 4, 8, 10], &[-1. / l, 0.5, 1. / l, 0.5], ks(s.iy) * l);
        }
    }
    if s.kappa <= 0. {
        // Euler-Bernoulli (legacy BEAM4 real-constant input)
        let (ez, ey) = (mat.e * s.iz, mat.e * s.iy);
        let l2 = l * l;
        let l3 = l2 * l;
        let pz = [
            (1usize, 1usize, 12. * ez / l3), (1, 5, 6. * ez / l2), (1, 7, -12. * ez / l3), (1, 11, 6. * ez / l2),
            (5, 5, 4. * ez / l), (5, 7, -6. * ez / l2), (5, 11, 2. * ez / l), (7, 7, 12. * ez / l3),
            (7, 11, -6. * ez / l2), (11, 11, 4. * ez / l),
        ];
        let py = [
            (2usize, 2usize, 12. * ey / l3), (2, 4, -6. * ey / l2), (2, 8, -12. * ey / l3), (2, 10, -6. * ey / l2),
            (4, 4, 4. * ey / l), (4, 8, 6. * ey / l2), (4, 10, 2. * ey / l), (8, 8, 12. * ey / l3),
            (8, 10, 6. * ey / l2), (10, 10, 4. * ey / l),
        ];
        for (i, j, v) in pz.iter().chain(py.iter()) {
            k[i * 12 + j] += v;
            if i != j {
                k[j * 12 + i] += v;
            }
        }
    }
    Ok(k)
}

// ---------------------------------------------------------------- public interface

pub fn elem_dof_mask(m: &Model, e: &Element) -> u8 {
    match m.kind_of(e) {
        Some(Kind::Spring) => match m.keyopt(e, 2) {
            1 => 0b001,
            2 => 0b010,
            3 => 0b100,
            _ => M_UXYZ,
        },
        Some(k) => k.dof_mask(),
        None => 0,
    }
}

/// nodes that carry DOFs for this element
pub fn elem_dof_nodes(m: &Model, e: &Element) -> Vec<usize> {
    match m.kind_of(e) {
        Some(Kind::Link) | Some(Kind::Spring) | Some(Kind::Beam) => e.nodes.iter().take(2).copied().collect(),
        Some(Kind::Mass21) => e.nodes.iter().take(1).copied().collect(),
        Some(Kind::Surf154) | None => vec![],
        Some(k) => match prepare(m, e, k) {
            Ok(c) => c.ids,
            Err(_) => e.nodes.iter().copied().filter(|&n| n > 0).collect(),
        },
    }
}

fn dofs_for(nodes: &[usize], mask: u8) -> Vec<(usize, u8)> {
    let mut v = Vec::new();
    for &n in nodes {
        for d in 0..6u8 {
            if mask & (1 << d) != 0 {
                v.push((n, d));
            }
        }
    }
    v
}

fn link_area(m: &Model, e: &Element) -> f64 {
    m.sections
        .get(&e.sec)
        .filter(|s| s.kind == "LINK")
        .and_then(|s| s.data.first().copied())
        .or_else(|| m.reals.get(&e.real).and_then(|r| r.first().copied()))
        .unwrap_or(0.)
}

fn two_nodes(m: &Model, e: &Element) -> Result<([f64; 3], [f64; 3]), String> {
    let i = *e.nodes.first().ok_or("element needs 2 nodes")?;
    let j = *e.nodes.get(1).ok_or("element needs 2 nodes")?;
    Ok((
        m.node(i).ok_or_else(|| format!("node {i} is not defined"))?,
        m.node(j).ok_or_else(|| format!("node {j} is not defined"))?,
    ))
}

pub fn element_matrix(m: &Model, e: &Element, acel: [f64; 3]) -> Result<Option<ElemMat>, String> {
    let kind = match m.kind_of(e) {
        Some(k) => k,
        None => return Ok(None),
    };
    match kind {
        Kind::Surf154 => Ok(None),
        Kind::Mass21 => {
            let r = m.reals.get(&e.real).cloned().unwrap_or_default();
            let m0 = r.first().copied().unwrap_or(0.);
            let mm = [m0, r.get(1).copied().unwrap_or(m0), r.get(2).copied().unwrap_or(m0)];
            let f = (0..3).map(|c| -acel[c] * mm[c]).collect();
            Ok(Some(ElemMat { dofs: dofs_for(&e.nodes[..1], M_UXYZ), k: vec![0.; 9], f }))
        }
        Kind::Link | Kind::Spring => {
            let (xi, xj) = two_nodes(m, e)?;
            if kind == Kind::Spring && m.keyopt(e, 2) >= 1 {
                let k = m.reals.get(&e.real).and_then(|r| r.first().copied()).unwrap_or(0.);
                return Ok(Some(ElemMat {
                    dofs: dofs_for(&e.nodes[..2], elem_dof_mask(m, e)),
                    k: vec![k, -k, -k, k],
                    f: vec![0.; 2],
                }));
            }
            let d = sub(xj, xi);
            let l = norm(d);
            if l <= 0. {
                return Err("link/spring element has zero length".into());
            }
            let c = unit(d);
            let (kax, fw) = if kind == Kind::Link {
                let mat = iso(m, e.mat)?;
                let a = link_area(m, e);
                if a <= 0. {
                    return Err("LINK180 has no cross-section area (SECTYPE,,LINK + SECDATA or real constant)".into());
                }
                (mat.e * a / l, mat.rho * a * l / 2.)
            } else {
                (m.reals.get(&e.real).and_then(|r| r.first().copied()).unwrap_or(0.), 0.)
            };
            let mut k = vec![0.; 36];
            for i in 0..3 {
                for j in 0..3 {
                    let v = kax * c[i] * c[j];
                    k[i * 6 + j] = v;
                    k[(i + 3) * 6 + j + 3] = v;
                    k[i * 6 + j + 3] = -v;
                    k[(i + 3) * 6 + j] = -v;
                }
            }
            let mut f = vec![0.; 6];
            for c in 0..3 {
                f[c] = -acel[c] * fw;
                f[c + 3] = -acel[c] * fw;
            }
            Ok(Some(ElemMat { dofs: dofs_for(&e.nodes[..2], M_UXYZ), k, f }))
        }
        Kind::Beam => {
            let (xi, xj) = two_nodes(m, e)?;
            let (r, l) = beam_frame(m, e, xi, xj)?;
            let kl = beam_local(m, e, l)?;
            let k = transform(&kl, &r, 12);
            let mat = iso(m, e.mat)?;
            let a = beam_section(m, e, mat.nu)?.a;
            let fw = mat.rho * a * l / 2.;
            let mut f = vec![0.; 12];
            for c in 0..3 {
                f[c] = -acel[c] * fw;
                f[c + 6] = -acel[c] * fw;
            }
            Ok(Some(ElemMat { dofs: dofs_for(&e.nodes[..2], M_ALL6), k, f }))
        }
        _ => {
            let c = prepare(m, e, kind)?;
            let (k, f, _) = c.stiffness(acel)?;
            let mask = if c.ndim() == 3 { M_UXYZ } else { M_UXY };
            Ok(Some(ElemMat { dofs: dofs_for(&c.ids, mask), k, f }))
        }
    }
}

/// Element results. `u(node, dof)` gives the solved displacement.
pub fn element_results(m: &Model, e: &Element, u: &dyn Fn(usize, u8) -> f64) -> Result<ElemOut, String> {
    let kind = match m.kind_of(e) {
        Some(k) => k,
        None => return Ok(ElemOut::default()),
    };
    match kind {
        Kind::Surf154 | Kind::Mass21 => Ok(ElemOut::default()),
        Kind::Link | Kind::Spring => {
            let (xi, xj) = two_nodes(m, e)?;
            if kind == Kind::Spring && m.keyopt(e, 2) >= 1 {
                let d = (m.keyopt(e, 2) - 1) as u8;
                let k = m.reals.get(&e.real).and_then(|r| r.first().copied()).unwrap_or(0.);
                return Ok(ElemOut { nodal: vec![], forces: vec![k * (u(e.nodes[1], d) - u(e.nodes[0], d))] });
            }
            let d = sub(xj, xi);
            let l = norm(d);
            let c = unit(d);
            let du: f64 = (0..3).map(|k| c[k] * (u(e.nodes[1], k as u8) - u(e.nodes[0], k as u8))).sum();
            if kind == Kind::Link {
                let mat = iso(m, e.mat)?;
                let a = link_area(m, e);
                let s = mat.e * du / l;
                Ok(ElemOut { nodal: vec![], forces: vec![s * a, s] })
            } else {
                let k = m.reals.get(&e.real).and_then(|r| r.first().copied()).unwrap_or(0.);
                Ok(ElemOut { nodal: vec![], forces: vec![k * du] })
            }
        }
        Kind::Beam => {
            let (xi, xj) = two_nodes(m, e)?;
            let (r, l) = beam_frame(m, e, xi, xj)?;
            let kl = beam_local(m, e, l)?;
            let mut ul = vec![0.; 12];
            for nd in 0..2 {
                for blk in 0..2 {
                    for i in 0..3 {
                        ul[nd * 6 + blk * 3 + i] = (0..3).map(|j| r[i][j] * u(e.nodes[nd], (blk * 3 + j) as u8)).sum();
                    }
                }
            }
            let f: Vec<f64> = (0..12).map(|i| (0..12).map(|j| kl[i * 12 + j] * ul[j]).sum()).collect();
            Ok(ElemOut { nodal: vec![], forces: f })
        }
        _ => {
            let c = prepare(m, e, kind)?;
            let nd = c.ndim();
            let mut ue = Vec::with_capacity(nd * c.ids.len());
            for &n in &c.ids {
                for d in 0..nd {
                    ue.push(u(n, d as u8));
                }
            }
            let s = c.stresses(&ue)?;
            let mut nodal: Vec<(usize, [f64; 6])> = Vec::with_capacity(c.ids.len());
            let mut cnt: Vec<usize> = Vec::new();
            for (i, &n) in c.ids.iter().enumerate() {
                if let Some(p) = nodal.iter().position(|x| x.0 == n) {
                    for q in 0..6 {
                        nodal[p].1[q] += s[i][q];
                    }
                    cnt[p] += 1;
                } else {
                    nodal.push((n, s[i]));
                    cnt.push(1);
                }
            }
            for (p, c) in cnt.iter().enumerate() {
                if *c > 1 {
                    for q in 0..6 {
                        nodal[p].1[q] /= *c as f64;
                    }
                }
            }
            Ok(ElemOut { nodal, forces: vec![] })
        }
    }
}

// ---------------------------------------------------------------- faces & pressure

/// faces of an element as indices into its node list (MAPDL face numbering)
pub fn faces(m: &Model, e: &Element) -> Vec<Vec<usize>> {
    let kind = match m.kind_of(e) {
        Some(k) => k,
        None => return vec![],
    };
    let n = e.nodes.len();
    match kind {
        Kind::Solid185 => vec![
            vec![1, 0, 3, 2], vec![0, 1, 5, 4], vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7], vec![4, 5, 6, 7],
        ],
        Kind::Solid186 if n >= 20 => vec![
            vec![1, 0, 3, 2, 8, 11, 10, 9],
            vec![0, 1, 5, 4, 8, 17, 12, 16],
            vec![1, 2, 6, 5, 9, 18, 13, 17],
            vec![2, 3, 7, 6, 10, 19, 14, 18],
            vec![3, 0, 4, 7, 11, 16, 15, 19],
            vec![4, 5, 6, 7, 12, 13, 14, 15],
        ],
        Kind::Solid187 if n >= 10 => vec![
            vec![1, 0, 2, 4, 6, 5], vec![0, 1, 3, 4, 8, 7], vec![1, 2, 3, 5, 9, 8], vec![2, 0, 3, 6, 7, 9],
        ],
        Kind::Plane182 => vec![vec![1, 0], vec![2, 1], vec![3, 2], vec![0, 3]],
        Kind::Plane183 if n >= 8 && e.nodes[6] != 0 => vec![vec![1, 0, 4], vec![2, 1, 5], vec![3, 2, 6], vec![0, 3, 7]],
        Kind::Plane183 if n >= 6 => vec![vec![1, 0, 3], vec![2, 1, 4], vec![0, 2, 5]],
        Kind::Surf154 => {
            let k = e.nodes.iter().filter(|&&x| x > 0).count();
            match k {
                3 => vec![vec![0, 1, 2]],
                6 => vec![vec![0, 1, 2, 3, 4, 5]],
                8 => vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
                _ => vec![vec![0, 1, 2, 3]],
            }
        }
        _ => vec![],
    }
}

/// Consistent nodal forces of pressure p on one face (face given as node ids).
/// Positive pressure pushes into the element. For SURF154 the element normal (I-J-K right hand) is used.
pub fn face_pressure(m: &Model, e: &Element, face: &[usize], p: f64) -> Result<Vec<(usize, [f64; 3])>, String> {
    let kind = m.kind_of(e).ok_or("unknown element type")?;
    let use_surface_normal = kind == Kind::Surf154;
    let x: Vec<[f64; 3]> =
        face.iter().map(|&n| m.node(n).ok_or_else(|| format!("node {n} missing"))).collect::<Result<_, _>>()?;
    let mut cen = [0.; 3];
    let mut cnt = 0.;
    for &n in &e.nodes {
        if let Some(c) = m.node(n) {
            for k in 0..3 {
                cen[k] += c[k];
            }
            cnt += 1.;
        }
    }
    for k in 0..3 {
        cen[k] /= f64::max(cnt, 1.);
    }
    let mut fc = [0.; 3];
    for c in &x {
        for k in 0..3 {
            fc[k] += c[k] / x.len() as f64;
        }
    }
    let out_dir = sub(fc, cen);
    let mut forces: Vec<(usize, [f64; 3])> = face.iter().map(|&n| (n, [0.; 3])).collect();
    if kind.is_plane() {
        let shp = if face.len() == 3 { Shape::Line3 } else { Shape::Line2 };
        let axi = m.keyopt(e, 3) == 1;
        let thick = if m.keyopt(e, 3) == 3 {
            m.reals.get(&e.real).and_then(|r| r.first().copied()).unwrap_or(1.)
        } else {
            1.
        };
        for (pt, w) in rule_line(3) {
            let (n, dn) = shp.eval(pt);
            let mut t = [0.; 3];
            let mut r = 0.;
            for a in 0..n.len() {
                for k in 0..2 {
                    t[k] += dn[a][0] * x[a][k];
                }
                r += n[a] * x[a][0];
            }
            let mut nrm = [t[1], -t[0], 0.];
            if dot(nrm, out_dir) < 0. {
                nrm = [-nrm[0], -nrm[1], 0.];
            }
            let fac = if axi { 2. * std::f64::consts::PI * r } else { thick };
            for a in 0..n.len() {
                for k in 0..2 {
                    forces[a].1[k] -= p * n[a] * nrm[k] * w * fac;
                }
            }
        }
        return Ok(forces);
    }
    let (shp, rule) = match face.len() {
        3 => (Shape::Tri3, rule_tri3()),
        6 => (Shape::Tri6, rule_tri3()),
        8 => (Shape::Quad8, rule_quad(3)),
        _ => (Shape::Quad4, rule_quad(2)),
    };
    for (pt, w) in rule {
        let (n, dn) = shp.eval(pt);
        let mut t1 = [0.; 3];
        let mut t2 = [0.; 3];
        for a in 0..n.len() {
            for k in 0..3 {
                t1[k] += dn[a][0] * x[a][k];
                t2[k] += dn[a][1] * x[a][k];
            }
        }
        let mut nrm = cross(t1, t2);
        if !use_surface_normal && dot(nrm, out_dir) < 0. {
            nrm = [-nrm[0], -nrm[1], -nrm[2]];
        }
        for a in 0..n.len() {
            for k in 0..3 {
                forces[a].1[k] -= p * n[a] * nrm[k] * w;
            }
        }
    }
    Ok(forces)
}

// ---------------------------------------------------------------- stress helpers

pub fn principal(s: &[f64; 6]) -> [f64; 3] {
    let (sx, sy, sz, sxy, syz, sxz) = (s[0], s[1], s[2], s[3], s[4], s[5]);
    let i1 = sx + sy + sz;
    let i2 = sx * sy + sy * sz + sz * sx - sxy * sxy - syz * syz - sxz * sxz;
    let i3 = sx * sy * sz + 2. * sxy * syz * sxz - sx * syz * syz - sy * sxz * sxz - sz * sxy * sxy;
    let q = (i1 * i1 - 3. * i2) / 9.;
    let r = -(2. * i1.powi(3) - 9. * i1 * i2 + 27. * i3) / 54.;
    if q <= 1e-300 {
        let m = i1 / 3.;
        return [m, m, m];
    }
    let th = (r / q.powf(1.5)).clamp(-1., 1.).acos();
    let sq = -2. * q.sqrt();
    let pi = std::f64::consts::PI;
    let mut v = [
        sq * (th / 3.).cos() + i1 / 3.,
        sq * ((th + 2. * pi) / 3.).cos() + i1 / 3.,
        sq * ((th - 2. * pi) / 3.).cos() + i1 / 3.,
    ];
    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
    v
}

pub fn von_mises(s: &[f64; 6]) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2))
        + 3. * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .sqrt()
}
