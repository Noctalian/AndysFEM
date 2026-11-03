//! Common result schema, cross-engine comparison and the verification ladder.

use super::adapters::Engine;
use super::jobs::Bundle;
use super::model::*;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct Effective {
    pub engine: Engine,
    pub engine_version: String,
    /// 2-D effective stiffness, Voigt (xx, yy, engineering xy)
    pub c: [[f64; 3]; 3],
    pub e1: f64,
    pub e2: f64,
    pub nu12: f64,
    pub nu21: f64,
    pub g12: f64,
    /// complete tensor only for the 3-case homogenisation load
    pub complete: bool,
    /// raw averaged stresses per strain case
    pub stress_averages: Vec<[f64; 3]>,
    pub strain_cases: Vec<[f64; 3]>,
    pub porosity: Option<f64>,
}

fn inv3(a: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-300 {
        return None;
    }
    let mut r = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2) = ((j + 1) % 3, (j + 2) % 3);
            let (j1, j2) = ((i + 1) % 3, (i + 2) % 3);
            r[i][j] = (a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]) / det;
        }
    }
    Some(r)
}

/// read results.json of a bundle and turn it into the common schema (also writes normalized.json)
pub fn effective(b: &Bundle) -> Option<Effective> {
    let txt = std::fs::read_to_string(b.dir.join("results.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let cases: Vec<[f64; 3]> = serde_json::from_value(v.get("strain_cases")?.clone()).ok()?;
    let avg: Vec<[f64; 3]> = serde_json::from_value(v.get("stress_averages")?.clone()).ok()?;
    let mut c = [[0.; 3]; 3];
    let complete = cases.len() == 3 && (0..3).all(|j| (0..3).all(|i| (i == j) == (cases[j][i] != 0.)));
    if complete {
        for j in 0..3 {
            for i in 0..3 {
                c[i][j] = avg[j][i] / cases[j][j];
            }
        }
    }
    let s = inv3(&c).unwrap_or([[f64::NAN; 3]; 3]);
    let porosity = std::fs::read_to_string(b.dir.join("geometry.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|g| g["porosity"].as_f64());
    let eff = Effective {
        engine: b.engine,
        engine_version: v["engine_version"].as_str().unwrap_or("unknown").to_string(),
        c,
        e1: 1. / s[0][0],
        e2: 1. / s[1][1],
        nu12: -s[0][1] / s[0][0],
        nu21: -s[0][1] / s[1][1],
        g12: 1. / s[2][2],
        complete,
        stress_averages: avg,
        strain_cases: cases,
        porosity,
    };
    let _ = std::fs::write(
        b.dir.join("normalized.json"),
        serde_json::to_string_pretty(&serde_json::json!({"schema": "fem-client/result@1", "case_id": b.case_id, "job": b.job, "effective": eff})).unwrap_or_default(),
    );
    Some(eff)
}

/// mixed strain/stress control from the effective tensor (exact for linear elasticity):
/// prescribe exx, keep syy = sxy = 0 -> eyy, gxy and sxx
pub fn uniaxial_response(e: &Effective, exx: f64) -> (f64, f64, f64) {
    let c = e.c;
    // [c11 c12 c13][exx]   [sxx]
    // [c21 c22 c23][eyy] = [ 0 ]
    // [c31 c32 c33][gxy]   [ 0 ]
    let det = c[1][1] * c[2][2] - c[1][2] * c[2][1];
    let eyy = -(c[1][0] * c[2][2] - c[1][2] * c[2][0]) / det * exx;
    let gxy = -(c[1][1] * c[2][0] - c[1][0] * c[2][1]) / det * exx;
    let sxx = c[0][0] * exx + c[0][1] * eyy + c[0][2] * gxy;
    (eyy, gxy, sxx)
}

pub struct Row {
    pub name: String,
    pub values: BTreeMap<Engine, f64>,
    pub max_rel_diff: f64,
    pub flagged: bool,
}

/// per-component comparison of the engines that finished the same case
pub fn cross_check(effs: &[Effective], tol: f64) -> Vec<Row> {
    let mut rows = Vec::new();
    if effs.is_empty() {
        return rows;
    }
    let scale = effs.iter().flat_map(|e| e.c.iter().flatten().map(|x| x.abs())).fold(0., f64::max).max(1e-300);
    let names = ["C11", "C12", "C13", "C21", "C22", "C23", "C31", "C32", "C33"];
    for k in 0..9 {
        let (i, j) = (k / 3, k % 3);
        let mut values = BTreeMap::new();
        for e in effs {
            values.insert(e.engine, e.c[i][j]);
        }
        let (mx, mn) = values.values().fold((f64::NEG_INFINITY, f64::INFINITY), |a, &v| (a.0.max(v), a.1.min(v)));
        // relative to the component, but components near zero are measured against the largest one
        let refv = values.values().map(|v| v.abs()).fold(0., f64::max).max(1e-6 * scale);
        let d = if values.len() > 1 { (mx - mn) / refv } else { 0. };
        rows.push(Row { name: names[k].into(), values, max_rel_diff: d, flagged: d > tol });
    }
    for (name, f) in [("nu12", 0usize), ("nu21", 1), ("E1", 2), ("E2", 3), ("G12", 4)] {
        let mut values = BTreeMap::new();
        for e in effs {
            values.insert(e.engine, [e.nu12, e.nu21, e.e1, e.e2, e.g12][f]);
        }
        let (mx, mn) = values.values().fold((f64::NEG_INFINITY, f64::INFINITY), |a, &v| (a.0.max(v), a.1.min(v)));
        let refv = values.values().map(|v| v.abs()).fold(0., f64::max).max(1e-12);
        let d = if values.len() > 1 { (mx - mn) / refv } else { 0. };
        rows.push(Row { name: name.into(), values, max_rel_diff: d, flagged: d > tol });
    }
    rows
}

pub struct Check {
    pub step: &'static str,
    pub name: String,
    pub passed: Option<bool>,
    pub detail: String,
}

/// verification ladder steps 1-3 that can be judged from one engine's result
pub fn ladder(job: &Job, e: &Effective, tol: f64) -> Vec<Check> {
    let mut v = Vec::new();
    let cm = job.material.plane_stress_c();
    if job.cell.kind == CellKind::Homogeneous {
        let mut worst: f64 = 0.;
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((e.c[i][j] - cm[i][j]).abs() / cm[0][0]);
            }
        }
        v.push(Check {
            step: "1 homogeneous cell",
            name: "returns the material's own plane-stress stiffness".into(),
            passed: Some(worst <= tol),
            detail: format!("max |C - C_mat| / C_mat11 = {:.3e}", worst),
        });
    }
    let solid = e.porosity.map(|p| 1. - p).unwrap_or_else(|| job.cell.solid_fraction());
    let voigt = [cm[0][0] * solid, cm[1][1] * solid, cm[2][2] * solid];
    let ok = e.c[0][0] <= voigt[0] * (1. + tol) && e.c[1][1] <= voigt[1] * (1. + tol) && e.c[2][2] <= voigt[2] * (1. + tol);
    v.push(Check {
        step: "2 analytical limits",
        name: "below the Voigt bound (1 - porosity) C_mat; Reuss bound of a porous cell is 0".into(),
        passed: Some(ok && e.c[0][0] >= 0. && e.c[1][1] >= 0. && e.c[2][2] >= 0.),
        detail: format!("C11 {:.4e} <= {:.4e}, C22 {:.4e} <= {:.4e}, C33 {:.4e} <= {:.4e} (solid fraction {:.4})",
            e.c[0][0], voigt[0], e.c[1][1], voigt[1], e.c[2][2], voigt[2], solid),
    });
    if job.cell.kind == CellKind::RotatingSquares {
        v.push(Check {
            step: "2 analytical limits",
            name: "ideal-hinge rotating squares tend to nu -> -1 as t -> 0".into(),
            passed: None,
            detail: format!("nu12 = {:.4} at t/a = {:.4} (judge the trend over a t sweep)", e.nu12, job.cell.t / job.cell.a),
        });
    }
    let sym = (e.c[0][1] - e.c[1][0]).abs() / e.c[0][0].abs().max(1e-300);
    v.push(Check {
        step: "3 consistency",
        name: "effective tensor symmetric (Hill-Mandel energy consistency)".into(),
        passed: Some(sym <= tol),
        detail: format!("|C12 - C21| / C11 = {:.3e}", sym),
    });
    v
}
