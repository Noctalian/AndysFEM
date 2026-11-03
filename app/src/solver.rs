//! Linear static solution: parallel assembly straight into a sparse pattern,
//! sparse direct (supernodal Cholesky) or PCG, then reactions and nodal stresses.

use crate::elements::{self, ElemMat};
use crate::model::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SolverKind {
    Sparse,
    Pcg,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub threads: usize,
    pub solver: SolverKind,
    pub pcg_tol: f64,
}

/// PCG default tolerance: the 1.0E-8 constant loaded by EQSLVE (0x14e4f4430, data 0x156426430)
/// and by the PCG driver (0x14ca51240, data 0x1562acf98).
pub const PCG_DEFAULT_TOL: f64 = 1.0e-8;

impl Default for Settings {
    fn default() -> Self {
        Settings { threads: 1, solver: SolverKind::Sparse, pcg_tol: PCG_DEFAULT_TOL }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Stats {
    pub equations: usize,
    pub nonzeros_lower: usize,
    pub factor_nonzeros: usize,
    pub solver: String,
    pub threads: usize,
    pub pcg_iterations: usize,
    pub pcg_achieved_tol: f64,
    pub t_setup: f64,
    pub t_assembly: f64,
    pub t_solve: f64,
    pub t_post: f64,
    pub t_total: f64,
    pub active_elements: usize,
    pub active_nodes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ResultSet {
    pub disp: Vec<[f64; 6]>, // by node id
    pub mask: Vec<u8>,       // dofs present per node id
    pub reac: BTreeMap<(usize, u8), f64>,
    pub stress: Vec<Option<[f64; 6]>>, // averaged nodal stress by node id
    pub elem_forces: BTreeMap<usize, Vec<f64>>,
    pub stats: Stats,
    pub warnings: Vec<String>,
}

impl ResultSet {
    pub fn u(&self, n: usize, d: u8) -> f64 {
        self.disp.get(n).map(|v| v[d as usize]).unwrap_or(0.)
    }
    pub fn usum(&self, n: usize) -> f64 {
        let u = self.disp.get(n).copied().unwrap_or([0.; 6]);
        (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()
    }
}

struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

fn atomic_add(a: &AtomicU64, v: f64) {
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        let new = (f64::from_bits(cur) + v).to_bits();
        match a.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => cur = x,
        }
    }
}

const NONE: u32 = u32::MAX;
const FIXED: u32 = u32::MAX - 1;

pub fn solve(m: &Model, s: &Settings, log: &mut dyn FnMut(String)) -> Result<ResultSet, String> {
    let threads = s.threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?;
    if threads > 1 {
        faer::set_global_parallelism(faer::Par::rayon(threads));
    } else {
        faer::set_global_parallelism(faer::Par::Seq);
    }
    let mut msgs = Vec::new();
    let r = pool.install(|| solve_inner(m, s, threads, &mut |x| msgs.push(x)));
    for x in msgs {
        log(x);
    }
    r
}

fn solve_inner(m: &Model, s: &Settings, threads: usize, log: &mut dyn FnMut(String)) -> Result<ResultSet, String> {
    let t0 = Instant::now();
    let mut warnings = Vec::new();
    // ------------------------------------------------ active elements and DOFs
    let elems: Vec<usize> = m.sel_elem_ids().into_iter().filter(|&i| m.kind_of(m.elems[i].as_ref().unwrap()).is_some()).collect();
    if elems.is_empty() {
        return Err("No elements are selected (or no element type is defined). Nothing to solve.".into());
    }
    let nmax = m.nodes.len();
    let mut mask = vec![0u8; nmax];
    let dof_nodes: Vec<Vec<usize>> = elems
        .par_iter()
        .map(|&e| elements::elem_dof_nodes(m, m.elems[e].as_ref().unwrap()))
        .collect();
    for (k, &e) in elems.iter().enumerate() {
        let em = elements::elem_dof_mask(m, m.elems[e].as_ref().unwrap());
        for &n in &dof_nodes[k] {
            if n >= nmax || m.nodes[n].is_none() {
                return Err(format!("Element {e} uses node {n}, which is not defined."));
            }
            mask[n] |= em;
        }
    }
    // compact node list (node id order first)
    let mut cnode: Vec<usize> = (0..nmax).filter(|&n| mask[n] != 0).collect();
    let mut cidx = vec![NONE; nmax];
    for (c, &n) in cnode.iter().enumerate() {
        cidx[n] = c as u32;
    }
    let ncn = cnode.len();
    // ------------------------------------------------ node adjacency
    let mut n2e_cnt = vec![0u32; ncn + 1];
    for dn in &dof_nodes {
        for &n in dn {
            n2e_cnt[cidx[n] as usize + 1] += 1;
        }
    }
    for i in 0..ncn {
        n2e_cnt[i + 1] += n2e_cnt[i];
    }
    let mut fillp = n2e_cnt.clone();
    let mut n2e = vec![0u32; n2e_cnt[ncn] as usize];
    for (k, dn) in dof_nodes.iter().enumerate() {
        for &n in dn {
            let c = cidx[n] as usize;
            n2e[fillp[c] as usize] = k as u32;
            fillp[c] += 1;
        }
    }
    let mut nbrs: Vec<Vec<u32>> = (0..ncn)
        .into_par_iter()
        .map(|c| {
            let mut v: Vec<u32> = Vec::new();
            for &k in &n2e[n2e_cnt[c] as usize..n2e_cnt[c + 1] as usize] {
                for &n in &dof_nodes[k as usize] {
                    v.push(cidx[n]);
                }
            }
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    // ------------------------------------------------ fill-reducing node order (geometric nested dissection)
    {
        let coords: Vec<[f64; 3]> = cnode.iter().map(|&n| m.nodes[n].unwrap()).collect();
        let order = nested_dissection(&coords, &nbrs);
        let mut newpos = vec![0u32; ncn];
        for (p, &o) in order.iter().enumerate() {
            newpos[o as usize] = p as u32;
        }
        let old_cnode = cnode.clone();
        let old_nbrs = std::mem::take(&mut nbrs);
        cnode = order.iter().map(|&o| old_cnode[o as usize]).collect();
        for (c, &n) in cnode.iter().enumerate() {
            cidx[n] = c as u32;
        }
        nbrs = order
            .par_iter()
            .map(|&o| {
                let mut v: Vec<u32> = old_nbrs[o as usize].iter().map(|&b| newpos[b as usize]).collect();
                v.sort_unstable();
                v
            })
            .collect();
    }
    // equation numbering in the new node order
    let mut eq = vec![NONE; nmax * 6];
    let mut node_eq_start: Vec<u32> = Vec::with_capacity(ncn + 1);
    let mut neq: u32 = 0;
    let mut nfixed = 0usize;
    for &n in &cnode {
        node_eq_start.push(neq);
        for d in 0..6u8 {
            if mask[n] & (1 << d) != 0 {
                if m.disp.contains_key(&(n, d)) {
                    eq[n * 6 + d as usize] = FIXED;
                    nfixed += 1;
                } else {
                    eq[n * 6 + d as usize] = neq;
                    neq += 1;
                }
            }
        }
    }
    node_eq_start.push(neq);
    let ignored = m.disp.keys().filter(|(n, _)| *n >= nmax || (mask[*n] == 0 && m.nsel.get(*n).copied().unwrap_or(false) && m.nodes[*n].is_some())).map(|k| k.0).collect::<std::collections::BTreeSet<_>>().len();
    if ignored > 0 {
        warnings.push(format!("{ignored} constrained node(s) are not attached to any selected element; their constraints were ignored."));
    }
    let neq = neq as usize;
    if neq == 0 {
        return Err("All degrees of freedom are constrained; nothing to solve.".into());
    }
    // ------------------------------------------------ sparse pattern
    let full = s.solver == SolverKind::Pcg;
    let node_free = |c: usize| -> std::ops::Range<u32> { node_eq_start[c]..node_eq_start[c + 1] };
    let mut row_cnt = vec![0u32; neq + 1];
    {
        let rc = SyncPtr(row_cnt.as_mut_ptr());
        (0..ncn).into_par_iter().for_each(|c| {
            let rc = &rc;
            for r in node_free(c) {
                let mut cnt = 0u32;
                for &b in &nbrs[c] {
                    let rg = node_free(b as usize);
                    if full {
                        cnt += rg.end - rg.start;
                    } else if rg.start <= r {
                        cnt += rg.end.min(r + 1) - rg.start;
                    }
                }
                unsafe { *rc.0.add(r as usize + 1) = cnt };
            }
        });
    }
    for i in 0..neq {
        let (a, b) = (row_cnt[i], row_cnt[i + 1]);
        row_cnt[i + 1] = a.checked_add(b).ok_or("model too large for 32-bit sparse indices")?;
    }
    let row_ptr = row_cnt;
    let nnz = row_ptr[neq] as usize;
    let mut cols = vec![0u32; nnz];
    {
        let cp = SyncPtr(cols.as_mut_ptr());
        (0..ncn).into_par_iter().for_each(|c| {
            let cp = &cp;
            for r in node_free(c) {
                let mut p = row_ptr[r as usize] as usize;
                for &b in &nbrs[c] {
                    for q in node_free(b as usize) {
                        if full || q <= r {
                            unsafe { *cp.0.add(p) = q };
                            p += 1;
                        }
                    }
                }
            }
        });
    }
    drop(nbrs);
    let t_setup = t0.elapsed().as_secs_f64();
    log(format!("   Equations: {neq}   Constrained DOFs: {nfixed}   Matrix terms ({}): {nnz}", if full { "full" } else { "lower" }));
    // ------------------------------------------------ assembly
    let t1 = Instant::now();
    let vals: Vec<AtomicU64> = (0..nnz).into_par_iter().map(|_| AtomicU64::new(0)).collect();
    let rhs: Vec<AtomicU64> = (0..neq).into_par_iter().map(|_| AtomicU64::new(0)).collect();
    let err: Mutex<Option<String>> = Mutex::new(None);
    let fixed_val = |n: usize, d: u8| m.disp.get(&(n, d)).copied().unwrap_or(0.);
    let pos = |r: u32, c: u32| -> usize {
        let (a, b) = (row_ptr[r as usize] as usize, row_ptr[r as usize + 1] as usize);
        a + cols[a..b].binary_search(&c).expect("pattern")
    };
    elems.par_iter().for_each(|&eid| {
        let e = m.elems[eid].as_ref().unwrap();
        let em: ElemMat = match elements::element_matrix(m, e, m.acel) {
            Ok(Some(x)) => x,
            Ok(None) => return,
            Err(msg) => {
                let mut g = err.lock().unwrap();
                if g.is_none() {
                    *g = Some(format!("Element {eid}: {msg}"));
                }
                return;
            }
        };
        let nd = em.dofs.len();
        let map: Vec<u32> = em.dofs.iter().map(|&(n, d)| eq[n * 6 + d as usize]).collect();
        for i in 0..nd {
            let ri = map[i];
            if ri == NONE || ri == FIXED {
                continue;
            }
            if em.f[i] != 0. {
                atomic_add(&rhs[ri as usize], em.f[i]);
            }
            for j in 0..nd {
                let kij = em.k[i * nd + j];
                if kij == 0. {
                    continue;
                }
                let cj = map[j];
                if cj == FIXED {
                    let (n, d) = em.dofs[j];
                    let uc = fixed_val(n, d);
                    if uc != 0. {
                        atomic_add(&rhs[ri as usize], -kij * uc);
                    }
                } else if cj != NONE && (full || cj <= ri) {
                    atomic_add(&vals[pos(ri, cj)], kij);
                }
            }
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    let vals: Vec<f64> = vals.into_par_iter().map(|a| f64::from_bits(a.into_inner())).collect();
    let mut b: Vec<f64> = rhs.into_par_iter().map(|a| f64::from_bits(a.into_inner())).collect();
    // external loads
    let mut fext: BTreeMap<(usize, u8), f64> = BTreeMap::new();
    for (&(n, d), &v) in &m.force {
        if n < nmax && mask[n] & (1 << d) != 0 {
            *fext.entry((n, d)).or_insert(0.) += v;
        } else {
            warnings.push(format!("Force {} on node {n} acts on a DOF no selected element uses; ignored.", FORCE_NAMES[d as usize]));
        }
    }
    for (n, d, v) in pressure_forces(m, &elems, &mut warnings)? {
        if mask[n] & (1 << d) != 0 {
            *fext.entry((n, d)).or_insert(0.) += v;
        }
    }
    for (&(n, d), &v) in &fext {
        let r = eq[n * 6 + d as usize];
        if r != NONE && r != FIXED {
            b[r as usize] += v;
        }
    }
    let t_assembly = t1.elapsed().as_secs_f64();
    // ------------------------------------------------ solve
    let t2 = Instant::now();
    let mut stats = Stats { equations: neq, nonzeros_lower: if full { (nnz + neq) / 2 } else { nnz }, threads, ..Default::default() };
    let mut solver = s.solver;
    if solver == SolverKind::Pcg && nfixed < 3 {
        warnings.push("The PCG solver only works for fully constrained models (fewer than 3 constraints found). The solver has been reset to SPARSE.".into());
        solver = SolverKind::Sparse;
    }
    let x: Vec<f64> = match solver {
        SolverKind::Sparse => {
            stats.solver = "SPARSE".into();
            let (rp, cl, vl);
            if full {
                // keep the lower part only (PCG fell back)
                let mut nrp = vec![0u32; neq + 1];
                let mut ncl = Vec::new();
                let mut nvl = Vec::new();
                for r in 0..neq {
                    for p in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                        if cols[p] as usize <= r {
                            ncl.push(cols[p]);
                            nvl.push(vals[p]);
                        }
                    }
                    nrp[r + 1] = ncl.len() as u32;
                }
                rp = nrp;
                cl = ncl;
                vl = nvl;
            } else {
                rp = row_ptr.clone();
                cl = std::mem::take(&mut cols);
                vl = vals;
            }
            let (x, fnz) = sparse_direct(neq, &rp, &cl, &vl, &b)?;
            stats.factor_nonzeros = fnz;
            log(format!("   Sparse factor terms: {fnz}"));
            x
        }
        SolverKind::Pcg => {
            stats.solver = "PCG".into();
            let maxit = (neq / 2).max(2000);
            let (x, it, tol) = pcg(neq, &row_ptr, &cols, &vals, &b, &node_eq_start, s.pcg_tol, maxit)?;
            stats.pcg_iterations = it;
            stats.pcg_achieved_tol = tol;
            log(format!("   Input PCG Error Tolerance: {:e}", s.pcg_tol));
            log(format!("   Achieved PCG Error Tolerance: {:e}   (iterations: {it})", tol));
            if tol > s.pcg_tol {
                return Err(format!(
                    "The PCG solver did not converge in {it} iterations (achieved {tol:e}). Check the model for rigid body motion or use EQSLV,SPARSE."
                ));
            }
            x
        }
    };
    let t_solve = t2.elapsed().as_secs_f64();
    // ------------------------------------------------ results
    let t3 = Instant::now();
    let mut disp = vec![[0.; 6]; nmax];
    for n in 0..nmax {
        if mask[n] == 0 {
            continue;
        }
        for d in 0..6u8 {
            match eq[n * 6 + d as usize] {
                NONE => {}
                FIXED => disp[n][d as usize] = fixed_val(n, d),
                r => disp[n][d as usize] = x[r as usize],
            }
        }
    }
    let ufn = |n: usize, d: u8| disp.get(n).map(|v| v[d as usize]).unwrap_or(0.);
    struct Post {
        nodal: Vec<(usize, [f64; 6])>,
        forces: Option<(usize, Vec<f64>)>,
        internal: Vec<((usize, u8), f64)>,
    }
    let posts: Vec<Result<Post, String>> = elems
        .par_iter()
        .map(|&eid| {
            let e = m.elems[eid].as_ref().unwrap();
            let out = elements::element_results(m, e, &ufn).map_err(|x| format!("Element {eid}: {x}"))?;
            let mut internal = Vec::new();
            if let Some(em) = elements::element_matrix(m, e, m.acel).map_err(|x| format!("Element {eid}: {x}"))? {
                let nd = em.dofs.len();
                for i in 0..nd {
                    let (n, d) = em.dofs[i];
                    if eq[n * 6 + d as usize] == FIXED {
                        let mut f = -em.f[i];
                        for j in 0..nd {
                            let (nj, dj) = em.dofs[j];
                            f += em.k[i * nd + j] * ufn(nj, dj);
                        }
                        internal.push(((n, d), f));
                    }
                }
            }
            let forces = if out.forces.is_empty() { None } else { Some((eid, out.forces)) };
            Ok(Post { nodal: out.nodal, forces, internal })
        })
        .collect();
    let mut reac: BTreeMap<(usize, u8), f64> = BTreeMap::new();
    for (&(n, d), _) in m.disp.iter() {
        if n < nmax && mask[n] & (1 << d) != 0 {
            reac.insert((n, d), 0.);
        }
    }
    let mut ssum = vec![[0.; 6]; nmax];
    let mut scnt = vec![0u32; nmax];
    let mut elem_forces = BTreeMap::new();
    for p in posts {
        let p = p?;
        for (n, sv) in p.nodal {
            for q in 0..6 {
                ssum[n][q] += sv[q];
            }
            scnt[n] += 1;
        }
        if let Some((e, f)) = p.forces {
            elem_forces.insert(e, f);
        }
        for (k, f) in p.internal {
            *reac.entry(k).or_insert(0.) += f;
        }
    }
    for (k, v) in reac.iter_mut() {
        if let Some(f) = fext.get(k) {
            *v -= f;
        }
    }
    let stress: Vec<Option<[f64; 6]>> = (0..nmax)
        .map(|n| {
            if scnt[n] == 0 {
                None
            } else {
                let c = scnt[n] as f64;
                let v = ssum[n];
                Some([v[0] / c, v[1] / c, v[2] / c, v[3] / c, v[4] / c, v[5] / c])
            }
        })
        .collect();
    stats.t_setup = t_setup;
    stats.t_assembly = t_assembly;
    stats.t_solve = t_solve;
    stats.t_post = t3.elapsed().as_secs_f64();
    stats.t_total = t0.elapsed().as_secs_f64();
    stats.active_elements = elems.len();
    stats.active_nodes = ncn;
    Ok(ResultSet { disp, mask, reac, stress, elem_forces, stats, warnings })
}

/// equivalent nodal forces from SF (nodal pressures) and SFE (element face pressures)
fn pressure_forces(m: &Model, elems: &[usize], warnings: &mut Vec<String>) -> Result<Vec<(usize, u8, f64)>, String> {
    let mut out = Vec::new();
    if m.node_pres.is_empty() && m.elem_pres.is_empty() {
        return Ok(out);
    }
    let selected: std::collections::HashSet<usize> = elems.iter().copied().collect();
    let mut applied = 0usize;
    // SF on nodes: faces whose nodes all carry a pressure
    if !m.node_pres.is_empty() {
        for &eid in elems {
            let e = m.elems[eid].as_ref().unwrap();
            let k = m.kind_of(e);
            if !matches!(k, Some(kk) if kk.is_solid() || kk.is_plane()) {
                continue;
            }
            for f in elements::faces(m, e) {
                let ids: Vec<usize> = f.iter().map(|&i| e.nodes[i]).collect();
                if ids.iter().all(|n| m.node_pres.contains_key(n)) {
                    let p = ids.iter().map(|n| m.node_pres[n]).sum::<f64>() / ids.len() as f64;
                    for (n, fv) in elements::face_pressure(m, e, &ids, p).map_err(|x| format!("Element {eid}: {x}"))? {
                        for d in 0..3u8 {
                            if fv[d as usize] != 0. {
                                out.push((n, d, fv[d as usize]));
                            }
                        }
                    }
                    applied += 1;
                }
            }
        }
    }
    for (&(eid, face), &p) in &m.elem_pres {
        let e = match m.elems.get(eid).and_then(|x| x.as_ref()) {
            Some(e) => e,
            None => continue,
        };
        let is_surf = m.kind_of(e) == Some(Kind::Surf154);
        if !selected.contains(&eid) && !is_surf {
            continue;
        }
        let fl = elements::faces(m, e);
        let idx = if is_surf { 0 } else { face.saturating_sub(1) };
        if let Some(f) = fl.get(idx) {
            let ids: Vec<usize> = f.iter().map(|&i| e.nodes[i]).filter(|&n| n > 0).collect();
            for (n, fv) in elements::face_pressure(m, e, &ids, p).map_err(|x| format!("Element {eid}: {x}"))? {
                for d in 0..3u8 {
                    if fv[d as usize] != 0. {
                        out.push((n, d, fv[d as usize]));
                    }
                }
            }
            applied += 1;
        } else {
            warnings.push(format!("SFE face {face} does not exist on element {eid}; ignored."));
        }
    }
    if applied == 0 {
        warnings.push("Pressures were defined but no element face carries them.".into());
    }
    Ok(out)
}

fn sparse_direct(n: usize, row_ptr: &[u32], cols: &[u32], vals: &[f64], b: &[f64]) -> Result<(Vec<f64>, usize), String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::linalg::cholesky::{factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering};
    use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
    // lower CSR == upper CSC; rows are already in nested-dissection order
    let sym = SymbolicSparseColMatRef::new_checked(n, n, row_ptr, None, cols);
    let mat = SparseColMatRef::new(sym, vals);
    let symbolic = factorize_symbolic_cholesky(sym, faer::Side::Upper, SymmetricOrdering::Identity, CholeskySymbolicParams::default())
        .map_err(|e| format!("sparse solver (symbolic): {e:?}"))?;
    let par = faer::get_global_parallelism();
    let mut lval: Vec<f64> = Vec::new();
    lval.try_reserve_exact(symbolic.len_val()).map_err(|_| "not enough memory for the sparse factor".to_string())?;
    lval.resize(symbolic.len_val(), 0.0f64);
    let mut mem = MemBuffer::try_new(symbolic.factorize_numeric_llt_scratch::<f64>(par, Default::default()))
        .map_err(|_| "not enough memory for the sparse solver workspace".to_string())?;
    let llt = symbolic
        .factorize_numeric_llt::<f64>(&mut lval, mat, faer::Side::Upper, Default::default(), par, MemStack::new(&mut mem), Default::default())
        .map_err(|_| {
            "The sparse solver found a zero or negative pivot: the model has rigid body motion (insufficient constraints) \
             or an element with no stiffness. Check the D commands and material properties."
                .to_string()
        })?;
    let mut rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let mut mem2 = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, par));
    llt.solve_in_place_with_conj(faer::Conj::No, rhs.as_mut(), par, MemStack::new(&mut mem2));
    let x: Vec<f64> = (0..n).map(|i| rhs[(i, 0)]).collect();
    if x.iter().any(|v| !v.is_finite()) {
        return Err("The sparse solver produced non-finite values (singular stiffness matrix).".into());
    }
    Ok((x, symbolic.len_val()))
}

/// Geometric nested dissection on the node graph: split along the longest axis,
/// the separator is the set of left nodes touching the right half; separators go last.
fn nested_dissection(coords: &[[f64; 3]], nbrs: &[Vec<u32>]) -> Vec<u32> {
    use std::sync::atomic::AtomicU32;
    let n = coords.len();
    let label: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let counter = AtomicU32::new(1);
    fn rec(ids: Vec<u32>, coords: &[[f64; 3]], nbrs: &[Vec<u32>], label: &[AtomicU32], counter: &AtomicU32) -> Vec<u32> {
        if ids.len() <= 64 {
            return ids;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &i in &ids {
            let c = coords[i as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap()).unwrap();
        let mut ids = ids;
        let mid = ids.len() / 2;
        ids.select_nth_unstable_by(mid, |&a, &b| {
            coords[a as usize][ax].partial_cmp(&coords[b as usize][ax]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        // nodes on the cut plane go to the right so that a single node layer can separate
        let cut = coords[ids[mid] as usize][ax];
        let (mut left, mut right): (Vec<u32>, Vec<u32>) = ids.iter().partition(|&&i| coords[i as usize][ax] < cut);
        if left.is_empty() || right.is_empty() {
            left = ids[..mid].to_vec();
            right = ids[mid..].to_vec();
        }
        drop(ids);
        let tl = counter.fetch_add(2, Ordering::Relaxed);
        let tr = tl + 1;
        for &l in &left {
            label[l as usize].store(tl, Ordering::Relaxed);
        }
        for &r in &right {
            label[r as usize].store(tr, Ordering::Relaxed);
        }
        let touch = |i: u32, t: u32| nbrs[i as usize].iter().any(|&b| label[b as usize].load(Ordering::Relaxed) == t);
        let nl = left.iter().filter(|&&l| touch(l, tr)).count();
        let nr = right.iter().filter(|&&r| touch(r, tl)).count();
        let mut sep = Vec::new();
        if nr <= nl {
            right.retain(|&r| {
                let t = touch(r, tl);
                if t {
                    sep.push(r);
                }
                !t
            });
        } else {
            left.retain(|&l| {
                let t = touch(l, tr);
                if t {
                    sep.push(l);
                }
                !t
            });
        }
        let (mut a, b) = if left.len() + right.len() > 20_000 {
            rayon::join(|| rec(left, coords, nbrs, label, counter), || rec(right, coords, nbrs, label, counter))
        } else {
            (rec(left, coords, nbrs, label, counter), rec(right, coords, nbrs, label, counter))
        };
        a.extend(b);
        a.extend(sep);
        a
    }
    rec((0..n as u32).collect(), coords, nbrs, &label, &counter)
}

#[allow(clippy::too_many_arguments)]
fn pcg(
    n: usize,
    rp: &[u32],
    cl: &[u32],
    vl: &[f64],
    b: &[f64],
    node_eq_start: &[u32],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize, f64), String> {
    // nodal block-Jacobi preconditioner
    let nblk = node_eq_start.len() - 1;
    let blocks: Vec<(usize, usize, Vec<f64>)> = (0..nblk)
        .into_par_iter()
        .filter_map(|c| {
            let (s, e) = (node_eq_start[c] as usize, node_eq_start[c + 1] as usize);
            if s == e {
                return None;
            }
            let k = e - s;
            let mut a = vec![0.; k * k];
            for r in s..e {
                for p in rp[r] as usize..rp[r + 1] as usize {
                    let c = cl[p] as usize;
                    if c >= s && c < e {
                        a[(r - s) * k + (c - s)] = vl[p];
                    }
                }
            }
            let inv = elements::inv_small(k, &a).unwrap_or_else(|| {
                let mut d = vec![0.; k * k];
                for i in 0..k {
                    d[i * k + i] = if a[i * k + i].abs() > 0. { 1. / a[i * k + i] } else { 1. };
                }
                d
            });
            Some((s, k, inv))
        })
        .collect();
    let precond = |r: &[f64], z: &mut [f64]| {
        let zp = SyncPtr(z.as_mut_ptr());
        blocks.par_iter().for_each(|(s, k, inv)| {
            let zp = &zp;
            for i in 0..*k {
                let mut v = 0.;
                for j in 0..*k {
                    v += inv[i * k + j] * r[s + j];
                }
                unsafe { *zp.0.add(s + i) = v };
            }
        });
    };
    const CH: usize = 512;
    let spmv = |x: &[f64], y: &mut [f64]| {
        y.par_chunks_mut(CH).enumerate().for_each(|(ci, ys)| {
            let r0 = ci * CH;
            for (k, yr) in ys.iter_mut().enumerate() {
                let r = r0 + k;
                let (a, b) = (rp[r] as usize, rp[r + 1] as usize);
                let mut s = 0.;
                for (v, c) in vl[a..b].iter().zip(&cl[a..b]) {
                    s += v * unsafe { *x.get_unchecked(*c as usize) };
                }
                *yr = s;
            }
        });
    };
    let dotp = |a: &[f64], c: &[f64]| -> f64 {
        a.par_chunks(4096).zip(c.par_chunks(4096)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>()).sum()
    };
    let bnorm = dotp(b, b).sqrt();
    let mut x = vec![0.; n];
    if bnorm == 0. {
        return Ok((x, 0, 0.));
    }
    let mut r = b.to_vec();
    let mut z = vec![0.; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut q = vec![0.; n];
    let mut rz = dotp(&r, &z);
    let mut rel = 1.;
    for it in 1..=maxit {
        spmv(&p, &mut q);
        let pq = dotp(&p, &q);
        if pq <= 0. {
            return Err("The PCG solver detects that the stiffness matrix is not positive definite (rigid body motion?). Use EQSLV,SPARSE to check.".into());
        }
        let alpha = rz / pq;
        x.par_iter_mut().zip(p.par_iter()).for_each(|(xi, pi)| *xi += alpha * pi);
        r.par_iter_mut().zip(q.par_iter()).for_each(|(ri, qi)| *ri -= alpha * qi);
        rel = dotp(&r, &r).sqrt() / bnorm;
        if rel <= tol {
            return Ok((x, it, rel));
        }
        precond(&r, &mut z);
        let rz_new = dotp(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        p.par_iter_mut().zip(z.par_iter()).for_each(|(pi, zi)| *pi = zi + beta * *pi);
    }
    Ok((x, maxit, rel))
}


/// MAPDL FSUM: sum over the selected nodes of the element nodal forces of the selected elements
/// (element loads minus K_e u_e; SURF154 contributes its pressure load), forces and moments about the origin.
/// Checked against the original: on support nodes this is minus the reaction.
pub fn fsum(m: &Model, r: &ResultSet) -> [f64; 6] {
    let ufn = |n: usize, d: u8| r.disp.get(n).map(|v| v[d as usize]).unwrap_or(0.);
    let sel_n = |n: usize| m.nsel.get(n).copied().unwrap_or(false);
    let elems = m.sel_elem_ids();
    let parts: Vec<Vec<(usize, u8, f64)>> = elems
        .par_iter()
        .map(|&eid| {
            let e = m.elems[eid].as_ref().unwrap();
            let mut out = Vec::new();
            if let Ok(Some(em)) = elements::element_matrix(m, e, m.acel) {
                let nd = em.dofs.len();
                for i in 0..nd {
                    let (n, d) = em.dofs[i];
                    if !sel_n(n) {
                        continue;
                    }
                    let mut f = em.f[i];
                    for j in 0..nd {
                        let (nj, dj) = em.dofs[j];
                        f -= em.k[i * nd + j] * ufn(nj, dj);
                    }
                    out.push((n, d, f));
                }
            }
            // pressure carried by this element
            let mut faces: Vec<(Vec<usize>, f64)> = Vec::new();
            let fl = elements::faces(m, e);
            let is_surf = m.kind_of(e) == Some(Kind::Surf154);
            for (&(ee, face), &p) in m.elem_pres.range((eid, 0)..=(eid, usize::MAX)) {
                let _ = ee;
                let idx = if is_surf { 0 } else { face.saturating_sub(1) };
                if let Some(f) = fl.get(idx) {
                    faces.push((f.iter().map(|&i| e.nodes[i]).filter(|&n| n > 0).collect(), p));
                }
            }
            if !m.node_pres.is_empty() && matches!(m.kind_of(e), Some(k) if k.is_solid() || k.is_plane()) {
                for f in &fl {
                    let ids: Vec<usize> = f.iter().map(|&i| e.nodes[i]).collect();
                    if ids.iter().all(|n| m.node_pres.contains_key(n)) {
                        let p = ids.iter().map(|n| m.node_pres[n]).sum::<f64>() / ids.len() as f64;
                        faces.push((ids, p));
                    }
                }
            }
            for (ids, p) in faces {
                if let Ok(fv) = elements::face_pressure(m, e, &ids, p) {
                    for (n, v) in fv {
                        if sel_n(n) {
                            for d in 0..3u8 {
                                if v[d as usize] != 0. {
                                    out.push((n, d, v[d as usize]));
                                }
                            }
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mut t = [0.; 6];
    for p in parts {
        for (n, d, f) in p {
            let x = m.node(n).unwrap_or([0.; 3]);
            match d {
                0 => {
                    t[0] += f;
                    t[4] += x[2] * f;
                    t[5] -= x[1] * f;
                }
                1 => {
                    t[1] += f;
                    t[3] -= x[2] * f;
                    t[5] += x[0] * f;
                }
                2 => {
                    t[2] += f;
                    t[3] += x[1] * f;
                    t[4] -= x[0] * f;
                }
                _ => t[d as usize] += f,
            }
        }
    }
    t
}
