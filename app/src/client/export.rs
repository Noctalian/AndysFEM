//! Running the code window on any solver: the APDL code is interpreted once (no solve) into the
//! model, then each adapter writes that model in its engine's own format, runs it, and the nodal
//! results (displacements, reactions, stresses where available) come back into one ResultSet.

use super::adapters::{Engine, EngineSettings};
use crate::apdl::Session;
use crate::elements::{self, Plane, Shape};
use crate::model::*;
use crate::solver::{ResultSet, Settings};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

struct Cells {
    /// node ids in export order (index = position)
    order: Vec<usize>,
    index: BTreeMap<usize, usize>,
    cells: Vec<(Shape, Vec<usize>, usize)>, // shape, node ids, material
    plane: Plane,
    thick: f64,
    gdim: usize,
}

fn collect(m: &Model) -> Result<Cells, String> {
    let mut cells = Vec::new();
    let mut plane = None;
    let mut thick = 1.0;
    let mut other = BTreeMap::new();
    for id in m.sel_elem_ids() {
        let e = m.elems[id].as_ref().unwrap();
        match m.kind_of(e) {
            Some(Kind::Surf154) => continue,
            Some(k) if k.is_solid() || k.is_plane() => {
                let (shape, ids, pl, t) = elements::continuum_cell(m, e).ok_or_else(|| format!("element {id} cannot be read"))?;
                let mut u = ids.clone();
                u.sort_unstable();
                u.dedup();
                if u.len() != ids.len() {
                    return Err(format!("element {id} is a collapsed (degenerate) brick; export it as tetrahedra or use the rebuilt solver"));
                }
                if let Some(p) = plane {
                    if p != pl {
                        return Err("the model mixes 3-D and 2-D (or different plane) elements; one kind per export".into());
                    }
                }
                plane = Some(pl);
                thick = t;
                cells.push((shape, ids, e.mat));
            }
            Some(k) => {
                *other.entry(format!("{k:?}")).or_insert(0) += 1;
            }
            None => {}
        }
    }
    if !other.is_empty() {
        return Err(format!(
            "exports to other solvers carry solid and plane elements only; this model also has {:?}. Use the rebuilt solver for it.",
            other
        ));
    }
    if cells.is_empty() {
        return Err("no selected solid or plane elements to export".into());
    }
    let mut order: Vec<usize> = cells.iter().flat_map(|c| c.1.iter().copied()).collect();
    order.sort_unstable();
    order.dedup();
    let index = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let plane = plane.unwrap();
    Ok(Cells { order, index, cells, plane, thick, gdim: if plane == Plane::Solid3D { 3 } else { 2 } })
}

/// force scaling between the rebuilt (Ansys) convention and a per-unit-thickness / per-radian one
fn force_scale(c: &Cells) -> f64 {
    match c.plane {
        Plane::Axi => 2.0 * std::f64::consts::PI,
        Plane::Stress => c.thick,
        _ => 1.0,
    }
}

fn mat_of(m: &Model, id: usize) -> (f64, f64, f64) {
    let e = m.mat_prop(id, "EX").unwrap_or(0.);
    let nu = m.mat_prop(id, "PRXY").or_else(|| m.mat_prop(id, "NUXY")).unwrap_or(0.3);
    (e, nu, m.mat_prop(id, "DENS").unwrap_or(0.))
}

/// pressure faces of the selected elements (SF on nodes and SFE), as (node ids, pressure)
fn pressure_faces(m: &Model) -> Vec<(Vec<usize>, f64)> {
    let mut out = Vec::new();
    for id in m.sel_elem_ids() {
        let e = m.elems[id].as_ref().unwrap();
        let fl = elements::faces(m, e);
        let surf = m.kind_of(e) == Some(Kind::Surf154);
        for (fi, f) in fl.iter().enumerate() {
            let ids: Vec<usize> = f.iter().map(|&i| e.nodes[i]).filter(|&n| n > 0).collect();
            if let Some(&p) = m.elem_pres.get(&(id, if surf { 1 } else { fi + 1 })) {
                out.push((ids.clone(), p));
                continue;
            }
            if !surf && !m.node_pres.is_empty() && ids.iter().all(|n| m.node_pres.contains_key(n)) {
                out.push((ids.clone(), ids.iter().map(|n| m.node_pres[n]).sum::<f64>() / ids.len() as f64));
            }
        }
    }
    out
}

const AX: [&str; 3] = ["DX", "DY", "DZ"];

// ------------------------------------------------------------------ Code_Aster

fn aster_name(s: Shape) -> &'static str {
    match s {
        Shape::Line2 => "SEG2",
        Shape::Line3 => "SEG3",
        Shape::Tri3 => "TRIA3",
        Shape::Tri6 => "TRIA6",
        Shape::Quad4 => "QUAD4",
        Shape::Quad8 => "QUAD8",
        Shape::Tet4 => "TETRA4",
        Shape::Tet10 => "TETRA10",
        Shape::Hex8 => "HEXA8",
        Shape::Hex20 => "HEXA20",
    }
}

fn aster_order(s: Shape, ids: &[usize]) -> Vec<usize> {
    if s == Shape::Hex20 {
        // Ansys: 8 corners, 4 bottom edges, 4 top edges, 4 vertical edges
        // Aster: 8 corners, 4 bottom edges, 4 vertical edges, 4 top edges
        let mut v = ids[..12].to_vec();
        v.extend_from_slice(&ids[16..20]);
        v.extend_from_slice(&ids[12..16]);
        return v;
    }
    ids.to_vec()
}

pub fn export_aster(m: &Model, dir: &Path) -> Result<(), String> {
    let c = collect(m)?;
    let fs = force_scale(&c);
    let mut mail = String::from("TITRE\nexported by fem-client from the code window\nFINSF\n");
    let _ = writeln!(mail, "{}", if c.gdim == 3 { "COOR_3D" } else { "COOR_2D" });
    for &n in &c.order {
        let x = m.node(n).unwrap();
        if c.gdim == 3 {
            let _ = writeln!(mail, "N{n} {:.15e} {:.15e} {:.15e}", x[0], x[1], x[2]);
        } else {
            let _ = writeln!(mail, "N{n} {:.15e} {:.15e}", x[0], x[1]);
        }
    }
    mail.push_str("FINSF\n");
    let mut by_shape: BTreeMap<&str, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    let mut mats: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, (s, ids, mat)) in c.cells.iter().enumerate() {
        by_shape.entry(aster_name(*s)).or_default().push((k + 1, aster_order(*s, ids)));
        mats.entry(*mat).or_default().push(k + 1);
    }
    // pressure faces become boundary cells
    let mut pres: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut pres_val: BTreeMap<String, f64> = BTreeMap::new();
    let mut next = c.cells.len() + 1;
    for (ids, p) in pressure_faces(m) {
        let s = match (c.gdim, ids.len()) {
            (2, 2) => Shape::Line2,
            (2, 3) => Shape::Line3,
            (3, 3) => Shape::Tri3,
            (3, 6) => Shape::Tri6,
            (3, 4) => Shape::Quad4,
            (3, 8) => Shape::Quad8,
            _ => continue,
        };
        let key = format!("PRES{}", pres_val.iter().position(|(_, v)| *v == p).unwrap_or(pres_val.len()) + 1);
        pres_val.insert(key.clone(), p);
        by_shape.entry(aster_name(s)).or_default().push((next, ids));
        pres.entry(key).or_default().push(next);
        next += 1;
    }
    for (name, cs) in &by_shape {
        let _ = writeln!(mail, "{name}");
        for (k, ids) in cs {
            let _ = writeln!(mail, "M{k} {}", ids.iter().map(|n| format!("N{n}")).collect::<Vec<_>>().join(" "));
        }
        mail.push_str("FINSF\n");
    }
    let mut group_ma = |name: &str, cells: &[usize], mail: &mut String| {
        let _ = writeln!(mail, "GROUP_MA\n{name}");
        for ch in cells.chunks(8) {
            let _ = writeln!(mail, "{}", ch.iter().map(|k| format!("M{k}")).collect::<Vec<_>>().join(" "));
        }
        mail.push_str("FINSF\n");
    };
    let solid: Vec<usize> = (1..=c.cells.len()).collect();
    group_ma("SOLID", &solid, &mut mail);
    for (mat, cs) in &mats {
        group_ma(&format!("MAT{mat}"), cs, &mut mail);
    }
    for (k, cs) in &pres {
        group_ma(k, cs, &mut mail);
    }
    // node groups: constraints grouped by identical (dof, value) sets, forces per node, CE nodes
    let mut dgroups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut dsets: BTreeMap<usize, Vec<(u8, f64)>> = BTreeMap::new();
    for (&(n, d), &v) in &m.disp {
        if (d as usize) < c.gdim && c.index.contains_key(&n) {
            dsets.entry(n).or_default().push((d, v));
        }
    }
    for (n, set) in &dsets {
        let key = set.iter().map(|(d, v)| format!("{}={:e}", AX[*d as usize], v)).collect::<Vec<_>>().join(",");
        dgroups.entry(key).or_default().push(*n);
    }
    let mut fnodes: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
    for (&(n, d), &v) in &m.force {
        if (d as usize) < c.gdim && c.index.contains_key(&n) {
            fnodes.entry(n).or_insert([0.; 3])[d as usize] += v / fs;
        }
    }
    let mut node_groups: Vec<(String, Vec<usize>)> = Vec::new();
    let dnames: Vec<(String, String)> = dgroups.keys().enumerate().map(|(i, k)| (format!("D{}", i + 1), k.clone())).collect();
    for (name, key) in &dnames {
        node_groups.push((name.clone(), dgroups[key].clone()));
    }
    for n in fnodes.keys() {
        node_groups.push((format!("F{n}"), vec![*n]));
    }
    let mut ce_nodes: Vec<usize> = m.ces.iter().flat_map(|ce| [ce.slave, ce.master]).filter(|&n| n > 0).collect();
    ce_nodes.sort_unstable();
    ce_nodes.dedup();
    for n in &ce_nodes {
        if !fnodes.contains_key(n) {
            node_groups.push((format!("N{n}G"), vec![*n]));
        }
    }
    for (name, ns) in &node_groups {
        let _ = writeln!(mail, "GROUP_NO\n{name}");
        for ch in ns.chunks(8) {
            let _ = writeln!(mail, "{}", ch.iter().map(|n| format!("N{n}")).collect::<Vec<_>>().join(" "));
        }
        mail.push_str("FINSF\n");
    }
    mail.push_str("FIN\n");
    // ---- command file
    let modl = match c.plane {
        Plane::Solid3D => "3D",
        Plane::Stress => "C_PLAN",
        Plane::Strain => "D_PLAN",
        Plane::Axi => "AXIS",
    };
    let mut comm = String::from("# Generated by fem-client from the code window (Code_Aster command file)\nimport json\n");
    comm += "DEBUT(LANG='EN', PAR_LOT='NON')\nmesh = LIRE_MAILLAGE(FORMAT='ASTER', UNITE=20)\n";
    let _ = writeln!(comm, "model = AFFE_MODELE(MAILLAGE=mesh, AFFE=_F(TOUT='OUI', PHENOMENE='MECANIQUE', MODELISATION='{modl}'))");
    let mut affe = Vec::new();
    for mat in mats.keys() {
        let (e, nu, rho) = mat_of(m, *mat);
        let _ = writeln!(comm, "m{mat} = DEFI_MATERIAU(ELAS=_F(E={e:e}, NU={nu}, RHO={rho:e}))");
        affe.push(format!("_F(GROUP_MA='MAT{mat}', MATER=m{mat})"));
    }
    let _ = writeln!(comm, "cmat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=({},))", affe.join(", "));
    let mut loads = Vec::new();
    for (name, key) in &dnames {
        let comps: Vec<String> = key.split(',').map(|kv| kv.to_string()).collect();
        let _ = writeln!(comm, "{} = AFFE_CHAR_MECA(MODELE=model, DDL_IMPO=_F(GROUP_NO='{}', {}))", name.to_lowercase(), name, comps.join(", "));
        loads.push(name.to_lowercase());
    }
    if !fnodes.is_empty() {
        let items: Vec<String> = fnodes
            .iter()
            .map(|(n, f)| {
                let mut s = format!("_F(GROUP_NO='F{n}'");
                for d in 0..c.gdim {
                    let _ = write!(s, ", F{}={:e}", ["X", "Y", "Z"][d], f[d]);
                }
                s + ")"
            })
            .collect();
        let _ = writeln!(comm, "fnod = AFFE_CHAR_MECA(MODELE=model, FORCE_NODALE=({},))", items.join(",\n    "));
        loads.push("fnod".into());
    }
    if !pres.is_empty() {
        let items: Vec<String> = pres_val.iter().map(|(k, p)| format!("_F(GROUP_MA='{k}', PRES={p:e})")).collect();
        let _ = writeln!(comm, "pres = AFFE_CHAR_MECA(MODELE=model, PRES_REP=({},))", items.join(", "));
        loads.push("pres".into());
    }
    let a = m.acel;
    let g = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if g > 0. {
        let _ = writeln!(comm, "grav = AFFE_CHAR_MECA(MODELE=model, PESANTEUR=_F(GRAVITE={g:e}, DIRECTION=({:e}, {:e}, {:e})))", -a[0] / g, -a[1] / g, -a[2] / g);
        loads.push("grav".into());
    }
    if !m.ces.is_empty() {
        let mut items = Vec::new();
        for ce in &m.ces {
            if (ce.dof as usize) >= c.gdim {
                continue;
            }
            let gs = |n: usize| if fnodes.contains_key(&n) { format!("F{n}") } else { format!("N{n}G") };
            if ce.master == 0 {
                items.push(format!("_F(GROUP_NO=('{}',), DDL=('{}',), COEF_MULT=(1.0,), COEF_IMPO={:e})", gs(ce.slave), AX[ce.dof as usize], ce.c));
            } else {
                items.push(format!(
                    "_F(GROUP_NO=('{}', '{}'), DDL=('{d}', '{d}'), COEF_MULT=(1.0, -1.0), COEF_IMPO={:e})",
                    gs(ce.slave),
                    gs(ce.master),
                    ce.c,
                    d = AX[ce.dof as usize]
                ));
            }
        }
        if !items.is_empty() {
            let _ = writeln!(comm, "ces = AFFE_CHAR_MECA(MODELE=model, LIAISON_DDL=({},))", items.join(",\n    "));
            loads.push("ces".into());
        }
    }
    if loads.is_empty() {
        return Err("the model has no constraints or loads to export".into());
    }
    let _ = writeln!(comm, "res = MECA_STATIQUE(MODELE=model, CHAM_MATER=cmat, EXCIT=({},))", loads.iter().map(|l| format!("_F(CHARGE={l})")).collect::<Vec<_>>().join(", "));
    let scmp: &[&str] = if c.gdim == 3 { &["SIXX", "SIYY", "SIZZ", "SIXY", "SIYZ", "SIXZ"] } else { &["SIXX", "SIYY", "SIZZ", "SIXY"] };
    let dcmp: Vec<&str> = AX[..c.gdim].to_vec();
    let _ = write!(
        comm,
        r#"res = CALC_CHAMP(reuse=res, RESULTAT=res, FORCE='REAC_NODA', CONTRAINTE=('SIGM_ELNO', 'SIGM_NOEU'))
def comps(champ, typ, names):
    f = CREA_CHAMP(TYPE_CHAM=typ, OPERATION='EXTR', RESULTAT=res, NOM_CHAM=champ, NUME_ORDRE=1)
    out = {{}}
    for n in names:
        c = f.EXTR_COMP(n, [], 1)
        out[n] = dict(zip([int(k) for k in c.noeud], [float(v) for v in c.valeurs]))
    return out
U = comps('DEPL', 'NOEU_DEPL_R', {dcmp:?})
R = comps('REAC_NODA', 'NOEU_DEPL_R', {dcmp:?})
S = comps('SIGM_NOEU', 'NOEU_SIEF_R', {scmp:?})
order = json.load(open('node_order.json'))
try:
    import code_aster
    ver = getattr(code_aster, '__version__', 'unknown')
except Exception:
    ver = 'unknown'
json.dump({{'engine': 'code_aster', 'engine_version': ver, 'force_scale': {fs:e},
           'nodes': order,
           'U': [[U[c].get(i + 1, 0.0) for c in {dcmp:?}] for i in range(len(order))],
           'RF': [[order[i], j, R[c].get(i + 1, 0.0)] for i in range(len(order)) for j, c in enumerate({dcmp:?})],
           'S': [[S[c].get(i + 1, 0.0) for c in {scmp:?}] for i in range(len(order))]}},
          open('results_fields.json', 'w'))
FIN()
"#
    );
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("mesh.mail"), mail).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("job.comm"), comm).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("job.export"),
        "P actions make_etude\nP version stable\nP mode interactif\nP ncpus 1\nP memory_limit 2048\nP time_limit 3600\nF comm job.comm D 1\nF mail mesh.mail D 20\nF mess job.mess R 6\nF resu job.resu R 8\n",
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(dir.join("node_order.json"), serde_json::to_string(&c.order).unwrap_or_default()).map_err(|e| e.to_string())?;
    Ok(())
}

// ------------------------------------------------------------------ FEniCSx

fn gmsh_order(s: Shape, ids: &[usize]) -> Vec<usize> {
    match s {
        // Ansys tet10 mids: 01 12 20 03 13 23 ; Gmsh: 01 12 20 03 23 13
        Shape::Tet10 => vec![ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[9], ids[8]],
        _ => ids.to_vec(),
    }
}

pub fn export_fenicsx(m: &Model, dir: &Path) -> Result<(), String> {
    let c = collect(m)?;
    let s0 = c.cells[0].0;
    if c.cells.iter().any(|x| x.0 != s0) {
        return Err("FEniCSx export needs one cell type in the whole mesh".into());
    }
    let (cell, deg, gm) = match s0 {
        Shape::Tri3 => ("triangle", 1, 2),
        Shape::Tri6 => ("triangle", 2, 9),
        Shape::Quad4 => ("quadrilateral", 1, 3),
        Shape::Tet4 => ("tetrahedron", 1, 4),
        Shape::Tet10 => ("tetrahedron", 2, 11),
        Shape::Hex8 => ("hexahedron", 1, 5),
        _ => return Err("FEniCSx export supports 3/6-node triangles, 4-node quads, 4/10-node tets and 8-node bricks (no 20-node serendipity bricks or 8-node quads)".into()),
    };
    if !m.ces.is_empty() {
        return Err("constraint equations (CE) are not exported to FEniCSx yet; use the rebuilt solver or Code_Aster".into());
    }
    let mats: Vec<usize> = {
        let mut v: Vec<usize> = c.cells.iter().map(|x| x.2).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if mats.len() > 1 {
        return Err("FEniCSx export supports one material".into());
    }
    let (e, nu, rho) = mat_of(m, mats[0]);
    let fs = force_scale(&c);
    let xs: Vec<Vec<f64>> = c.order.iter().map(|n| m.node(*n).unwrap()[..c.gdim].to_vec()).collect();
    let cells: Vec<Vec<usize>> = c.cells.iter().map(|x| gmsh_order(x.0, &x.1).iter().map(|n| c.index[n]).collect()).collect();
    let bcs: Vec<(usize, u8, f64)> = m.disp.iter().filter(|((n, d), _)| (*d as usize) < c.gdim && c.index.contains_key(n)).map(|(&(n, d), &v)| (c.index[&n], d, v)).collect();
    let fls: Vec<(usize, u8, f64)> = m.force.iter().filter(|((n, d), _)| (*d as usize) < c.gdim && c.index.contains_key(n)).map(|(&(n, d), &v)| (c.index[&n], d, v / fs)).collect();
    if !pressure_faces(m).is_empty() {
        return Err("pressures are not exported to FEniCSx yet; convert them to nodal forces or use Code_Aster / the rebuilt solver".into());
    }
    let data = serde_json::json!({"x": xs, "cells": cells, "bcs": bcs, "forces": fls, "order": c.order,
        "E": e, "nu": nu, "rho": rho, "acel": m.acel, "plane": format!("{:?}", c.plane), "cell": cell, "degree": deg,
        "gmsh_type": gm, "force_scale": fs});
    let py = r#"# Generated by fem-client from the code window (FEniCSx / DOLFINx 0.11).
import json
import numpy as np
import ufl, basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem
import dolfinx.fem.petsc as fpetsc
D = json.load(open("model.json"))
gdim = len(D["x"][0])
x = np.array(D["x"], dtype=np.float64)
cells = np.array(D["cells"], dtype=np.int64)
try:
    from dolfinx.io.gmsh import cell_perm_array
except ImportError:
    from dolfinx.io.gmshio import cell_perm_array
perm = cell_perm_array(dolfinx.mesh.to_type(D["cell"]), cells.shape[1])
cells = cells[:, perm]
coord_el = basix.ufl.element("Lagrange", D["cell"], D["degree"], shape=(gdim,))
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, coord_el)
V = fem.functionspace(mesh, ("Lagrange", D["degree"], (gdim,)))
E, nu = D["E"], D["nu"]
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
if D["plane"] == "Stress":
    lmbda = E * nu / (1 - nu ** 2)
def eps(w):
    return ufl.sym(ufl.grad(w))
def sigma(w):
    return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2 * mu * eps(w)
# geometry node (input order) -> blocked dof, cell by cell (same element on geometry and V)
inp = mesh.geometry.input_global_indices
gdm = mesh.geometry.dofmap
node_dof = {}
for cc in range(gdm.shape[0]):
    cd = V.dofmap.cell_dofs(cc)
    for k, g in enumerate(gdm[cc]):
        node_dof[int(inp[g])] = int(cd[k])
bs = V.dofmap.index_map_bs
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
r = 1.0
if D["plane"] == "Axi":
    raise SystemExit("axisymmetric models are not exported to FEniCSx")
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
body = fem.Constant(mesh, np.array([-D["rho"] * D["acel"][k] for k in range(gdim)]))
L = ufl.inner(body, v) * ufl.dx
bcs = []
groups = {}
for (n, d, val) in D["bcs"]:
    groups.setdefault((d, val), []).append(node_dof[n] * bs + d)
for (d, val), dofs in groups.items():
    bcs.append(fem.dirichletbc(PETSc.ScalarType(val), np.array(dofs, dtype=np.int32), V.sub(d)))
af, lf = fem.form(a), fem.form(L)
A = fpetsc.assemble_matrix(af, bcs=bcs); A.assemble()
b = fpetsc.assemble_vector(lf)
F = np.zeros(b.getSize())
for (n, d, val) in D["forces"]:
    F[node_dof[n] * bs + d] += val
b.array[:] += F[: len(b.array)]
fpetsc.apply_lifting(b, [af], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fpetsc.set_bc(b, bcs)
ksp = PETSc.KSP().create(mesh.comm); ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu")
uh = fem.Function(V)
ksp.solve(b, uh.x.petsc_vec); uh.x.scatter_forward()
# reactions: K u - f at the constrained dofs (unconstrained assembly)
A0 = fpetsc.assemble_matrix(af); A0.assemble()
b0 = fpetsc.assemble_vector(lf)
Ku = A0.createVecLeft(); A0.mult(uh.x.petsc_vec, Ku)
res = Ku.array - b0.array - F[: len(b0.array)]
U, RF = [], []
for i, n in enumerate(D["order"]):
    dof = node_dof[i]
    U.append([float(uh.x.array[dof * bs + k]) for k in range(gdim)])
for (n, d, val) in D["bcs"]:
    RF.append([D["order"][n], d, float(res[node_dof[n] * bs + d])])
json.dump({"engine": "fenicsx", "engine_version": "dolfinx " + dolfinx.__version__, "force_scale": D["force_scale"],
           "nodes": D["order"], "U": U, "RF": RF, "S": None}, open("results_fields.json", "w"))
"#;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("model.json"), serde_json::to_string(&data).unwrap_or_default()).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("job_fenicsx.py"), py).map_err(|e| e.to_string())?;
    Ok(())
}

// ------------------------------------------------------------------ MAPDL

pub fn export_mapdl(code: &str, dir: &Path) -> Result<(), String> {
    let py = r#"# Generated by fem-client: runs the code window's APDL deck on the school MAPDL container (PyMAPDL)
import json, os, re
import numpy as np
from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl(start_instance=False, ip=os.environ.get("MAPDL_IP", "127.0.0.1"), port=int(os.environ.get("MAPDL_PORT", "50052")))
mapdl.clear()
mapdl.upload("model.dat")
mapdl.input("model.dat")
mapdl.finish(); mapdl.post1(); mapdl.set("LAST"); mapdl.allsel()
nnum = [int(n) for n in mapdl.mesh.nnum]
U = mapdl.post_processing.nodal_displacement("ALL").tolist()
try:
    S = np.column_stack([mapdl.post_processing.nodal_component_stress(c) for c in ("X", "Y", "Z", "XY", "YZ", "XZ")]).tolist()
except Exception:
    S = None
RF = []
txt = str(mapdl.prrsol())
cols = []
for line in txt.splitlines():
    w = line.split()
    if w and w[0] == "NODE":
        cols = w[1:]
    elif w and w[0].isdigit() and cols:
        for lab, val in zip(cols, w[1:]):
            if lab in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
                RF.append([int(w[0]), ["FX", "FY", "FZ", "MX", "MY", "MZ"].index(lab), float(val)])
json.dump({"engine": "mapdl", "engine_version": str(mapdl.version), "force_scale": 1.0,
           "nodes": nnum, "U": U, "RF": RF, "S": S}, open("results_fields.json", "w"))
mapdl.exit()
"#;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("model.dat"), code).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("job_mapdl.py"), py).map_err(|e| e.to_string())?;
    Ok(())
}

fn run_sh(engine: Engine, s: &EngineSettings) -> String {
    let mut out = String::from("#!/usr/bin/env bash\n# Generated by fem-client (code window run)\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n");
    match engine {
        Engine::CodeAster => out += &format!("{} run --rm -v \"$PWD\":/work:Z -w /work {} run_aster job.export\n", s.runtime, s.aster_image),
        Engine::Fenicsx => out += &format!("{} run --rm -v \"$PWD\":/work:Z -w /work {} python3 job_fenicsx.py\n", s.runtime, s.fenicsx_image),
        Engine::Mapdl => {
            out += &format!("export MAPDL_IP='{}' MAPDL_PORT='{}'\n", s.mapdl_ip, s.mapdl_port);
            out += &format!("uv run --no-project --with '{}' --with numpy python job_mapdl.py\n", s.pymapdl_pin);
        }
        Engine::Local => {}
    }
    out += "test -f results_fields.json\n";
    out
}

/// read results_fields.json into a ResultSet for the given model
pub fn read_fields(m: &Model, dir: &Path) -> Result<ResultSet, String> {
    let txt = std::fs::read_to_string(dir.join("results_fields.json")).map_err(|e| format!("no results_fields.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
    let nodes: Vec<usize> = serde_json::from_value(v["nodes"].clone()).map_err(|e| e.to_string())?;
    let u: Vec<Vec<f64>> = serde_json::from_value(v["U"].clone()).map_err(|e| e.to_string())?;
    let fs = v["force_scale"].as_f64().unwrap_or(1.0);
    let nmax = m.nodes.len().max(nodes.iter().max().map(|x| x + 1).unwrap_or(0));
    let mut r = ResultSet { disp: vec![[0.; 6]; nmax], mask: vec![0; nmax], stress: vec![None; nmax], ..Default::default() };
    for (k, &n) in nodes.iter().enumerate() {
        if let Some(row) = u.get(k) {
            for (d, val) in row.iter().enumerate().take(6) {
                r.disp[n][d] = *val;
                r.mask[n] |= 1 << d;
            }
        }
    }
    if let Some(rf) = v["RF"].as_array() {
        for t in rf {
            if let (Some(n), Some(d), Some(val)) = (t[0].as_u64(), t[1].as_u64(), t[2].as_f64()) {
                *r.reac.entry((n as usize, d as u8)).or_insert(0.) += val * fs;
            }
        }
    }
    if let Some(s) = v["S"].as_array() {
        for (k, row) in s.iter().enumerate() {
            if let (Some(&n), Some(row)) = (nodes.get(k), row.as_array()) {
                let mut st = [0.; 6];
                for (i, x) in row.iter().enumerate().take(6) {
                    st[i] = x.as_f64().unwrap_or(0.);
                }
                r.stress[n] = Some(st);
            }
        }
    }
    r.stats.solver = v["engine"].as_str().unwrap_or("?").to_string();
    r.stats.equations = nodes.len() * 3;
    r.warnings.push(format!("results from {} ({})", v["engine"].as_str().unwrap_or("?"), v["engine_version"].as_str().unwrap_or("?")));
    Ok(r)
}

/// interpret the code without solving and write the engine's job (no run); returns the job folder
pub fn prepare(code: &str, engine: Engine, s: &EngineSettings, workspace: &Path) -> Result<PathBuf, String> {
    let mut ses = Session::new(Settings::default());
    ses.no_solve = true;
    ses.run_text(code);
    if !ses.errors.is_empty() {
        return Err(ses.errors.join(" | "));
    }
    let model = ses.solve_models.first().cloned().ok_or("the code has no SOLVE command")?;
    let hash = super::sha256_hex(code.as_bytes())[..12].to_string();
    let dir = workspace.join("code_runs").join(&hash).join(engine.key());
    match engine {
        Engine::CodeAster => export_aster(&model, &dir)?,
        Engine::Fenicsx => export_fenicsx(&model, &dir)?,
        Engine::Mapdl => export_mapdl(code, &dir)?,
        Engine::Local => return Err("the rebuilt solver runs the code directly".into()),
    }
    std::fs::write(dir.join("run.sh"), run_sh(engine, s)).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// interpret the code without solving, export to the engine, run it, read the fields back.
/// Returns the session (model, listing) with the engine's result attached as the current result.
pub fn run_on_engine(code: &str, engine: Engine, s: &EngineSettings, workspace: &Path) -> Session {
    let mut ses = Session::new(Settings::default());
    ses.no_solve = true;
    ses.run_text(code);
    let hash = super::sha256_hex(code.as_bytes())[..12].to_string();
    let dir: PathBuf = workspace.join("code_runs").join(&hash).join(engine.key());
    let model = match ses.solve_models.first() {
        Some(m) => m.clone(),
        None => {
            ses.out.push_str("\n *** ERROR ***\n The code has no SOLVE command, so there is nothing to send to the solver.\n");
            ses.errors.push("no SOLVE".into());
            return ses;
        }
    };
    if ses.solve_models.len() > 1 {
        ses.out.push_str("\n *** NOTE ***  Only the first load step (first SOLVE) is sent to the external solver.\n");
    }
    let exp = match engine {
        Engine::CodeAster => export_aster(&model, &dir),
        Engine::Fenicsx => export_fenicsx(&model, &dir),
        Engine::Mapdl => export_mapdl(code, &dir),
        Engine::Local => Err("the rebuilt solver runs directly".into()),
    };
    if let Err(e) = exp {
        ses.out.push_str(&format!("\n *** ERROR ***\n Export to {}: {e}\n", engine.label()));
        ses.errors.push(e);
        return ses;
    }
    let _ = std::fs::write(dir.join("run.sh"), run_sh(engine, s));
    ses.out.push_str(&format!("\n Job written to {}\n Running {} …\n", dir.display(), engine.label()));
    let _ = std::fs::remove_file(dir.join("results_fields.json"));
    match std::process::Command::new("bash").arg("run.sh").current_dir(&dir).output() {
        Ok(o) => {
            let log = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
            let _ = std::fs::write(dir.join("log.txt"), &log);
            let tail: Vec<&str> = log.lines().rev().take(12).collect();
            ses.out.push_str(&format!(" --- engine log (last lines) ---\n{}\n", tail.into_iter().rev().collect::<Vec<_>>().join("\n")));
        }
        Err(e) => {
            ses.out.push_str(&format!("\n *** ERROR ***\n Cannot start bash ({e}); external solvers need Linux with uv and a container runtime.\n"));
            ses.errors.push("no bash".into());
            return ses;
        }
    }
    match read_fields(&model, &dir) {
        Ok(r) => {
            ses.model = model;
            ses.results.push(r);
            ses.cur = Some(ses.results.len() - 1);
            ses.out.push_str(" Results read back from the solver.\n");
        }
        Err(e) => {
            ses.out.push_str(&format!("\n *** ERROR ***\n {} did not produce results: {e}\n", engine.label()));
            ses.errors.push(e);
            ses.model = model;
        }
    }
    ses
}
