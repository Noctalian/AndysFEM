//! MAPDL-style listings and machine-readable exports (JSON / CSV).

use crate::elements::{principal, von_mises};
use crate::model::*;
use crate::solver::ResultSet;
use std::fmt::Write;

/// Fortran G-like number with 5 significant digits, 12 characters wide.
pub fn g(v: f64) -> String {
    let a = v.abs();
    let s = if v == 0.0 {
        "0.0000".to_string()
    } else if (0.1..99999.5).contains(&a) {
        let ar: f64 = format!("{:.4e}", a).parse().unwrap_or(a);
        let ndig = if ar < 1.0 { 0 } else { (ar.log10().floor() as i32) + 1 };
        let dec = (5 - ndig).max(0) as usize;
        format!("{:.*}", dec, v)
    } else {
        let mut e = a.log10().floor() as i32 + 1;
        let mut mant = a / 10f64.powi(e);
        if (mant * 1e5).round() >= 1e5 {
            e += 1;
            mant = a / 10f64.powi(e);
        }
        let sign = if v < 0.0 { "-" } else { "" };
        format!("{sign}{:.5}E{}{:02}", mant, if e < 0 { "-" } else { "+" }, e.abs())
    };
    format!("{:>12}", s)
}

fn header(out: &mut String, title: &str, what: &str) {
    let _ = writeln!(out, "\n PRINT {title}\n\n  ***** POST1 {what} *****\n\n  LOAD STEP=     1  SUBSTEP=     1\n   TIME=    1.0000      LOAD CASE=   0\n");
}

pub fn node_value(m: &Model, r: &ResultSet, n: usize, item: &str, comp: &str) -> Option<f64> {
    let item = item.to_uppercase();
    let comp = comp.to_uppercase();
    match item.as_str() {
        "U" => {
            if n >= r.mask.len() || r.mask[n] == 0 {
                return if m.node(n).is_some() { Some(0.) } else { None };
            }
            Some(match comp.as_str() {
                "X" => r.u(n, 0),
                "Y" => r.u(n, 1),
                "Z" => r.u(n, 2),
                _ => r.usum(n),
            })
        }
        "ROT" => {
            let v = r.disp.get(n).copied().unwrap_or([0.; 6]);
            Some(match comp.as_str() {
                "X" => v[3],
                "Y" => v[4],
                "Z" => v[5],
                _ => (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt(),
            })
        }
        "S" => {
            let s = r.stress.get(n).copied().flatten()?;
            let p = principal(&s);
            Some(match comp.as_str() {
                "X" => s[0],
                "Y" => s[1],
                "Z" => s[2],
                "XY" => s[3],
                "YZ" => s[4],
                "XZ" => s[5],
                "1" => p[0],
                "2" => p[1],
                "3" => p[2],
                "INT" => p[0] - p[2],
                _ => von_mises(&s),
            })
        }
        "RF" => {
            let d = FORCE_NAMES.iter().position(|f| *f == comp).unwrap_or(0) as u8;
            r.reac.get(&(n, d)).copied()
        }
        "LOC" => {
            let x = m.node(n)?;
            Some(match comp.as_str() {
                "Y" => x[1],
                "Z" => x[2],
                _ => x[0],
            })
        }
        _ => None,
    }
}

pub fn prnsol(m: &Model, r: &ResultSet, item: &str, comp: &str) -> String {
    let mut out = String::new();
    let item = item.to_uppercase();
    let comp = comp.to_uppercase();
    let (cols, title, what): (Vec<(&str, &str, &str)>, &str, &str) = match (item.as_str(), comp.as_str()) {
        ("U", "X") | ("U", "Y") | ("U", "Z") | ("U", "SUM") => (
            vec![(Box::leak(format!("U{comp}").into_boxed_str()) as &str, "U", Box::leak(comp.clone().into_boxed_str()) as &str)],
            "U    NODAL SOLUTION PER NODE",
            "NODAL DEGREE OF FREEDOM LISTING",
        ),
        ("U", _) => (
            vec![("UX", "U", "X"), ("UY", "U", "Y"), ("UZ", "U", "Z"), ("USUM", "U", "SUM")],
            "U    NODAL SOLUTION PER NODE",
            "NODAL DEGREE OF FREEDOM LISTING",
        ),
        ("ROT", _) => (
            vec![("ROTX", "ROT", "X"), ("ROTY", "ROT", "Y"), ("ROTZ", "ROT", "Z"), ("RSUM", "ROT", "SUM")],
            "ROT  NODAL SOLUTION PER NODE",
            "NODAL DEGREE OF FREEDOM LISTING",
        ),
        ("S", "PRIN") => (
            vec![("S1", "S", "1"), ("S2", "S", "2"), ("S3", "S", "3"), ("SINT", "S", "INT"), ("SEQV", "S", "EQV")],
            "S    NODAL SOLUTION PER NODE",
            "ELEMENT NODAL STRESS LISTING",
        ),
        ("S", "EQV") => (vec![("SEQV", "S", "EQV")], "S    NODAL SOLUTION PER NODE", "ELEMENT NODAL STRESS LISTING"),
        ("S", c) if !c.is_empty() && c != "COMP" => (
            vec![(Box::leak(format!("S{c}").into_boxed_str()) as &str, "S", Box::leak(c.to_string().into_boxed_str()) as &str)],
            "S    NODAL SOLUTION PER NODE",
            "ELEMENT NODAL STRESS LISTING",
        ),
        ("S", _) => (
            vec![("SX", "S", "X"), ("SY", "S", "Y"), ("SZ", "S", "Z"), ("SXY", "S", "XY"), ("SYZ", "S", "YZ"), ("SXZ", "S", "XZ")],
            "S    NODAL SOLUTION PER NODE",
            "ELEMENT NODAL STRESS LISTING",
        ),
        _ => {
            let _ = writeln!(out, " *** WARNING ***  PRNSOL,{item},{comp} is not available in this version (U, ROT, S are).");
            return out;
        }
    };
    header(&mut out, title, what);
    if item != "S" {
        let _ = writeln!(out, "  THE FOLLOWING DEGREE OF FREEDOM RESULTS ARE IN THE GLOBAL COORDINATE SYSTEM\n");
    } else {
        let _ = writeln!(out, "  THE FOLLOWING NODAL RESULTS ARE AVERAGED OVER THE SELECTED ELEMENTS\n");
    }
    let mut line = String::from("    NODE");
    for c in &cols {
        let _ = write!(line, "{:>13}", c.0);
    }
    let _ = writeln!(out, "{line}");
    let mut maxv = vec![(0usize, 0f64); cols.len()];
    for n in m.sel_node_ids() {
        let vals: Vec<Option<f64>> = cols.iter().map(|c| node_value(m, r, n, c.1, c.2)).collect();
        if vals.iter().all(|v| v.is_none()) {
            continue;
        }
        let mut l = format!("{:>8}", n);
        for (k, v) in vals.iter().enumerate() {
            let v = v.unwrap_or(0.);
            let _ = write!(l, " {}", g(v));
            if v.abs() > maxv[k].1.abs() {
                {
                    maxv[k] = (n, v);
                }
            }
        }
        let _ = writeln!(out, "{l}");
    }
    let _ = writeln!(out, "\n MAXIMUM ABSOLUTE VALUES");
    let mut l1 = String::from(" NODE   ");
    let mut l2 = String::from(" VALUE  ");
    for (n, v) in &maxv {
        let _ = write!(l1, " {:>12}", n);
        let _ = write!(l2, " {}", g(*v));
    }
    let _ = writeln!(out, "{l1}\n{l2}");
    out
}

pub fn prrsol(m: &Model, r: &ResultSet) -> String {
    let mut out = String::new();
    header(&mut out, "REACTION SOLUTIONS PER NODE", "TOTAL REACTION SOLUTION LISTING");
    let _ = writeln!(out, "  THE FOLLOWING X,Y,Z SOLUTIONS ARE IN THE GLOBAL COORDINATE SYSTEM\n");
    let used: Vec<u8> = (0..6u8).filter(|d| r.reac.keys().any(|k| k.1 == *d)).collect();
    let mut line = String::from("    NODE");
    for d in &used {
        let _ = write!(line, "{:>13}", FORCE_NAMES[*d as usize]);
    }
    let _ = writeln!(out, "{line}");
    let mut tot = vec![0.; used.len()];
    let mut nodes: Vec<usize> = r.reac.keys().map(|k| k.0).collect();
    nodes.dedup();
    for n in nodes {
        if !m.nsel.get(n).copied().unwrap_or(false) {
            continue;
        }
        let mut l = format!("{:>8}", n);
        for (k, d) in used.iter().enumerate() {
            match r.reac.get(&(n, *d)) {
                Some(v) => {
                    tot[k] += v;
                    let _ = write!(l, " {}", g(*v));
                }
                None => {
                    let _ = write!(l, " {:>12}", "");
                }
            }
        }
        let _ = writeln!(out, "{l}");
    }
    let mut l = String::from("\n TOTAL VALUES\n VALUE  ");
    for v in tot {
        let _ = write!(l, " {}", g(v));
    }
    let _ = writeln!(out, "{l}");
    out
}

pub fn presol(m: &Model, r: &ResultSet) -> String {
    let mut out = String::new();
    header(&mut out, "ELEMENT FORCES (LINK / BEAM / SPRING)", "ELEMENT SOLUTION LISTING");
    for (&e, f) in &r.elem_forces {
        if !m.esel.get(e).copied().unwrap_or(false) {
            continue;
        }
        let el = m.elems[e].as_ref().unwrap();
        match m.kind_of(el) {
            Some(Kind::Link) => {
                let _ = writeln!(out, " ELEM {:>8}  LINK180   AXIAL FORCE= {}  AXIAL STRESS= {}", e, g(f[0]), g(f[1]));
            }
            Some(Kind::Spring) => {
                let _ = writeln!(out, " ELEM {:>8}  COMBIN14  SPRING FORCE= {}", e, g(f[0]));
            }
            Some(Kind::Beam) => {
                let _ = writeln!(out, " ELEM {:>8}  BEAM188  (element coordinate system)", e);
                let _ = writeln!(out, "      NODE           FX           FY           FZ           MX           MY           MZ");
                for (k, n) in el.nodes.iter().take(2).enumerate() {
                    let s = if k == 0 { -1. } else { 1. };
                    let mut l = format!("  {:>8}", n);
                    for q in 0..6 {
                        // internal force convention: I end reversed
                        let _ = write!(l, " {}", g(s * f[k * 6 + q] * if k == 0 { -1. } else { 1. }));
                    }
                    let _ = writeln!(out, "{l}");
                }
            }
            _ => {}
        }
    }
    out
}

pub fn fsum(m: &Model, r: &ResultSet) -> ([f64; 6], String) {
    let t = crate::solver::fsum(m, r);
    let mut out = String::from("\n ***** SUMMATION OF TOTAL    FORCES AND MOMENTS IN THE GLOBAL COORDINATE SYSTEM *****\n");
    let _ = writeln!(out, "  FX  = {}\n  FY  = {}\n  FZ  = {}", g(t[0]), g(t[1]), g(t[2]));
    let _ = writeln!(out, "  MX  = {}\n  MY  = {}\n  MZ  = {}", g(t[3]), g(t[4]), g(t[5]));
    let _ = writeln!(out, " SUMMATION POINT=  0.0000      0.0000      0.0000");
    (t, out)
}

#[derive(serde::Serialize)]
struct JNode {
    id: usize,
    x: f64,
    y: f64,
    z: f64,
    ux: f64,
    uy: f64,
    uz: f64,
    usum: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    rot: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stress: Option<[f64; 6]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seqv: Option<f64>,
}

#[derive(serde::Serialize)]
struct JReac {
    node: usize,
    dof: &'static str,
    value: f64,
}

pub fn to_json(m: &Model, r: &ResultSet, jobname: &str, title: &str, warnings: &[String]) -> String {
    let mut nodes = Vec::new();
    let mut maxu = (0usize, 0f64);
    let mut maxs = (0usize, 0f64);
    for n in m.node_ids() {
        if n >= r.mask.len() || r.mask[n] == 0 {
            continue;
        }
        let x = m.node(n).unwrap();
        let d = r.disp[n];
        let us = r.usum(n);
        if us >= maxu.1 {
            maxu = (n, us);
        }
        let st = r.stress[n];
        let sq = st.map(|s| von_mises(&s));
        if let Some(v) = sq {
            if v >= maxs.1 {
                maxs = (n, v);
            }
        }
        nodes.push(JNode {
            id: n,
            x: x[0],
            y: x[1],
            z: x[2],
            ux: d[0],
            uy: d[1],
            uz: d[2],
            usum: us,
            rot: if r.mask[n] & 0b111000 != 0 { Some([d[3], d[4], d[5]]) } else { None },
            stress: st,
            seqv: sq,
        });
    }
    let reac: Vec<JReac> = r.reac.iter().map(|(&(n, d), &v)| JReac { node: n, dof: FORCE_NAMES[d as usize], value: v }).collect();
    let mut tot = [0.; 6];
    for (&(_, d), &v) in &r.reac {
        tot[d as usize] += v;
    }
    let ef: Vec<serde_json::Value> = r.elem_forces.iter().map(|(e, f)| serde_json::json!({"elem": e, "values": f})).collect();
    let v = serde_json::json!({
        "jobname": jobname,
        "title": title,
        "analysis": "linear static",
        "max_displacement": {"node": maxu.0, "usum": maxu.1},
        "max_von_mises": {"node": maxs.0, "seqv": maxs.1},
        "reaction_totals": {"FX": tot[0], "FY": tot[1], "FZ": tot[2], "MX": tot[3], "MY": tot[4], "MZ": tot[5]},
        "stats": r.stats,
        "warnings": warnings,
        "nodes": nodes,
        "reactions": reac,
        "element_forces": ef,
    });
    serde_json::to_string_pretty(&v).unwrap_or_default()
}

pub fn to_csv(m: &Model, r: &ResultSet) -> String {
    let mut s = String::from("node,x,y,z,ux,uy,uz,usum,sx,sy,sz,sxy,syz,sxz,seqv\n");
    for n in m.node_ids() {
        if n >= r.mask.len() || r.mask[n] == 0 {
            continue;
        }
        let x = m.node(n).unwrap();
        let d = r.disp[n];
        let _ = write!(s, "{n},{},{},{},{:e},{:e},{:e},{:e}", x[0], x[1], x[2], d[0], d[1], d[2], r.usum(n));
        match r.stress[n] {
            Some(st) => {
                let _ = writeln!(s, ",{:e},{:e},{:e},{:e},{:e},{:e},{:e}", st[0], st[1], st[2], st[3], st[4], st[5], von_mises(&st));
            }
            None => s.push_str(",,,,,,,\n"),
        }
    }
    s
}
