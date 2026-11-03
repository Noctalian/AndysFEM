//! MAPDL (APDL) command interpreter: parameters, *DO / *IF, PREP7, SOLU, POST1.

use crate::expr::{self, EvalCtx};
use crate::model::*;
use crate::post;
use crate::solver::{self, ResultSet, Settings, SolverKind};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Param {
    Num(f64),
    Str(String),
    Arr { d: [usize; 3], v: Vec<f64> },
}

enum Frame {
    Loop { var: String, end: f64, inc: f64, start: usize },
    If { taken: bool },
}

pub struct Session {
    pub model: Model,
    pub params: BTreeMap<String, Param>,
    pub out: String,
    pub settings: Settings,
    pub results: Vec<ResultSet>,
    pub cur: Option<usize>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub unsupported: BTreeMap<String, usize>,
    pub degrees: bool,
    pub processor: String,
    pub nopr: bool,
    pub dir: PathBuf,
    pub jobname: String,
    pub title: String,
    pub stop: bool,
    pub cancel: Arc<AtomicBool>,
    /// per result set: constraint-equation forces (force applied on the slave node)
    pub ce_forces: Vec<Vec<(Ce, f64)>>,
    /// build the model but do not solve (used to export the code to other solvers)
    pub no_solve: bool,
    /// model state at each SOLVE when `no_solve` is set
    pub solve_models: Vec<Model>,
    pub lines_run: usize,
    pub commands_run: usize,
    last_plnsol: (f64, f64, usize, usize),
    last_fsum: [f64; 6],
    mids: HashMap<(usize, usize), usize>,
    cf: Option<std::fs::File>,
    depth: usize,
}

struct Ctx<'a>(&'a Session);

impl<'a> EvalCtx for Ctx<'a> {
    fn var(&self, name: &str) -> Option<f64> {
        match self.0.params.get(name) {
            Some(Param::Num(v)) => Some(*v),
            Some(Param::Arr { v, .. }) => v.first().copied(),
            _ => None,
        }
    }
    fn degrees(&self) -> bool {
        self.0.degrees
    }
    fn call(&self, name: &str, a: &[f64]) -> Option<f64> {
        let s = self.0;
        if let Some(Param::Arr { d, v }) = s.params.get(name) {
            let i = a.first().copied().unwrap_or(1.) as usize;
            let j = a.get(1).copied().unwrap_or(1.) as usize;
            let k = a.get(2).copied().unwrap_or(1.) as usize;
            if i < 1 || j < 1 || k < 1 || i > d[0] || j > d[1] || k > d[2] {
                return None;
            }
            return v.get((i - 1) + (j - 1) * d[0] + (k - 1) * d[0] * d[1]).copied();
        }
        let m = &s.model;
        let n = a.first().copied().unwrap_or(0.) as usize;
        let res = s.cur.and_then(|c| s.results.get(c));
        let u = |d: u8| res.map(|r| r.u(n, d)).unwrap_or(0.);
        Some(match name {
            "UX" => u(0),
            "UY" => u(1),
            "UZ" => u(2),
            "ROTX" => u(3),
            "ROTY" => u(4),
            "ROTZ" => u(5),
            "NX" => m.node(n)?[0],
            "NY" => m.node(n)?[1],
            "NZ" => m.node(n)?[2],
            "NSEL" => match m.node(n) {
                Some(_) => {
                    if m.nsel[n] {
                        1.
                    } else {
                        -1.
                    }
                }
                None => 0.,
            },
            "ESEL" => match m.elems.get(n).and_then(|e| e.as_ref()) {
                Some(_) => {
                    if m.esel[n] {
                        1.
                    } else {
                        -1.
                    }
                }
                None => 0.,
            },
            "NDNEXT" => m.sel_node_ids().into_iter().find(|&k| k > n).unwrap_or(0) as f64,
            "ELNEXT" => m.sel_elem_ids().into_iter().find(|&k| k > n).unwrap_or(0) as f64,
            "NODE" => {
                let p = [a.first().copied()?, a.get(1).copied().unwrap_or(0.), a.get(2).copied().unwrap_or(0.)];
                let mut best = (0usize, f64::INFINITY);
                for id in m.sel_node_ids() {
                    let x = m.node(id).unwrap();
                    let d = (x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2) + (x[2] - p[2]).powi(2);
                    if d < best.1 {
                        best = (id, d);
                    }
                }
                best.0 as f64
            }
            "DISTND" => {
                let (p, q) = (m.node(n)?, m.node(a.get(1).copied()? as usize)?);
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
            }
            _ => return None,
        })
    }
}

fn split_fields(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut q = false;
    for c in s.chars() {
        match c {
            '\'' => {
                q = !q;
                cur.push(c);
            }
            '(' if !q => {
                depth += 1;
                cur.push(c);
            }
            ')' if !q => {
                depth -= 1;
                cur.push(c);
            }
            ',' if !q && depth <= 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn strip_comment(s: &str) -> &str {
    let mut q = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' => q = !q,
            '!' if !q => return &s[..i],
            _ => {}
        }
    }
    s
}

fn split_dollar(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut q = false;
    let mut st = 0;
    for (i, c) in s.char_indices() {
        match c {
            '\'' => q = !q,
            '$' if !q => {
                out.push(s[st..i].to_string());
                st = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[st..].to_string());
    out
}

fn dof_of(lab: &str) -> Option<u8> {
    DOF_NAMES.iter().position(|d| *d == lab).map(|p| p as u8)
}
fn force_of(lab: &str) -> Option<u8> {
    FORCE_NAMES.iter().position(|d| *d == lab).map(|p| p as u8)
}

/// parse a Fortran format like (3i9,6e21.13e3) into field widths with kinds (true = int)
fn fortran_widths(f: &str) -> Vec<(bool, usize)> {
    let f = f.trim().trim_start_matches('(').trim_end_matches(')').to_lowercase();
    let mut out = Vec::new();
    for part in f.split(',') {
        let part = part.trim();
        let pos = part.find(|c: char| c.is_alphabetic());
        let Some(pos) = pos else { continue };
        let rep: usize = if pos == 0 { 1 } else { part[..pos].parse().unwrap_or(1) };
        let kind = part.as_bytes()[pos] as char;
        let rest = &part[pos + 1..];
        let w: usize = rest.split(|c: char| !c.is_ascii_digit()).next().unwrap_or("0").parse().unwrap_or(0);
        if w == 0 {
            continue;
        }
        for _ in 0..rep {
            out.push((kind == 'i', w));
        }
    }
    out
}

fn fixed_fields(line: &str, widths: &[(bool, usize)]) -> Vec<String> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut p = 0;
    for &(_, w) in widths {
        if p >= b.len() {
            break;
        }
        let e = (p + w).min(b.len());
        out.push(String::from_utf8_lossy(&b[p..e]).trim().to_string());
        p = e;
    }
    out
}

fn pnum(s: &str) -> f64 {
    s.trim().replace(['d', 'D'], "e").parse().unwrap_or(0.)
}

impl Session {
    pub fn new(settings: Settings) -> Self {
        Session {
            model: Model::new(),
            params: BTreeMap::new(),
            out: String::new(),
            settings,
            results: Vec::new(),
            cur: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            unsupported: BTreeMap::new(),
            degrees: false,
            processor: "BEGIN".into(),
            nopr: false,
            dir: std::env::current_dir().unwrap_or_default(),
            jobname: "file".into(),
            title: String::new(),
            stop: false,
            cancel: Arc::new(AtomicBool::new(false)),
            ce_forces: Vec::new(),
            no_solve: false,
            solve_models: Vec::new(),
            lines_run: 0,
            commands_run: 0,
            last_plnsol: (0., 0., 0, 0),
            last_fsum: [0.; 6],
            mids: HashMap::new(),
            cf: None,
            depth: 0,
        }
    }

    pub fn result(&self) -> Option<&ResultSet> {
        self.cur.and_then(|c| self.results.get(c))
    }

    fn echo(&mut self, s: impl AsRef<str>) {
        if !self.nopr {
            self.out.push_str(s.as_ref());
            self.out.push('\n');
        }
    }
    fn print(&mut self, s: impl AsRef<str>) {
        self.out.push_str(s.as_ref());
        if !s.as_ref().ends_with('\n') {
            self.out.push('\n');
        }
    }
    fn warn(&mut self, s: impl Into<String>) {
        let s = s.into();
        let _ = writeln!(self.out, "\n *** WARNING ***\n {s}");
        self.warnings.push(s);
    }
    fn error(&mut self, s: impl Into<String>) {
        let s = s.into();
        let _ = writeln!(self.out, "\n *** ERROR ***\n {s}");
        self.errors.push(s);
    }

    fn eval(&self, s: &str) -> Result<f64, String> {
        expr::eval(s, &Ctx(self))
    }

    fn num(&mut self, f: &[String], i: usize, def: f64) -> f64 {
        match f.get(i) {
            None => def,
            Some(s) if s.is_empty() => def,
            Some(s) => {
                let s = s.clone();
                match self.eval(&s) {
                    Ok(v) => v,
                    Err(e) => {
                        self.error(format!("Cannot evaluate '{s}': {e}"));
                        def
                    }
                }
            }
        }
    }
    fn txt(f: &[String], i: usize) -> String {
        f.get(i).map(|s| s.trim().trim_matches('\'').to_uppercase()).unwrap_or_default()
    }

    /// %name% substitution
    fn subst(&self, line: &str) -> String {
        if !line.contains('%') {
            return line.to_string();
        }
        let mut out = String::new();
        let mut rest = line;
        while let Some(a) = rest.find('%') {
            out.push_str(&rest[..a]);
            let after = &rest[a + 1..];
            if let Some(b) = after.find('%') {
                let name = after[..b].trim().to_uppercase();
                let rep = match self.params.get(&name) {
                    Some(Param::Str(s)) => Some(s.clone()),
                    Some(Param::Num(v)) => Some(fmt_num(*v)),
                    _ => self.eval(&name).ok().map(fmt_num),
                };
                match rep {
                    Some(r) => {
                        out.push_str(&r);
                        rest = &after[b + 1..];
                    }
                    None => {
                        out.push('%');
                        rest = after;
                    }
                }
            } else {
                out.push('%');
                rest = after;
            }
        }
        out.push_str(rest);
        out
    }

    pub fn run_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        self.run_text(&text);
        Ok(())
    }

    pub fn run_text(&mut self, text: &str) {
        let mut prog: Vec<String> = Vec::new();
        for raw in text.lines() {
            let l = raw.trim_end_matches('\r');
            for p in split_dollar(strip_comment(l)) {
                prog.push(p);
            }
        }
        self.depth += 1;
        if self.depth > 20 {
            self.error("/INPUT nesting deeper than 20 levels");
            self.depth -= 1;
            return;
        }
        self.exec(&prog);
        self.depth -= 1;
    }

    fn find_block_end(prog: &[String], from: usize, open: &[&str], close: &[&str], stop_at: &[&str]) -> Option<usize> {
        let mut depth = 0;
        for (i, l) in prog.iter().enumerate().skip(from) {
            let u = l.trim().to_uppercase();
            let cmd = u.split(',').next().unwrap_or("").trim().to_string();
            let is_open = open.iter().any(|o| cmd == *o) && (open[0] != "*IF" || u.trim_end().ends_with("THEN"));
            if is_open {
                depth += 1;
                continue;
            }
            if depth == 0 && (close.iter().any(|c| cmd == *c) || stop_at.iter().any(|c| cmd == *c)) {
                return Some(i);
            }
            if close.iter().any(|c| cmd == *c) {
                depth -= 1;
            }
        }
        None
    }

    fn cond(&mut self, f: &[String]) -> bool {
        let op = Self::txt(f, 2);
        let (sa, sb) = (f.get(1).cloned().unwrap_or_default(), f.get(3).cloned().unwrap_or_default());
        let is_str = |s: &str| s.trim().starts_with('\'');
        if is_str(&sa) || is_str(&sb) {
            let get = |s: &str, me: &Session| -> String {
                let t = s.trim();
                if t.starts_with('\'') {
                    t.trim_matches('\'').to_uppercase()
                } else {
                    match me.params.get(&t.to_uppercase()) {
                        Some(Param::Str(x)) => x.to_uppercase(),
                        _ => t.to_uppercase(),
                    }
                }
            };
            let (a, b) = (get(&sa, self), get(&sb, self));
            return match op.as_str() {
                "EQ" => a == b,
                "NE" => a != b,
                _ => false,
            };
        }
        let a = self.num(f, 1, 0.);
        let b = self.num(f, 3, 0.);
        let tol = 1e-10;
        match op.as_str() {
            "EQ" => (a - b).abs() <= tol,
            "NE" => (a - b).abs() > tol,
            "LT" => a < b,
            "GT" => a > b,
            "LE" => a <= b,
            "GE" => a >= b,
            "ABLT" => a.abs() < b.abs(),
            "ABGT" => a.abs() > b.abs(),
            _ => {
                self.error(format!("*IF operator {op} is not valid"));
                false
            }
        }
    }

    fn exec(&mut self, prog: &[String]) {
        let mut pc = 0usize;
        let mut stack: Vec<Frame> = Vec::new();
        let mut guard: u64 = 0;
        while pc < prog.len() && !self.stop {
            if self.cancel.load(Ordering::Relaxed) {
                self.error("Run cancelled by the user.");
                self.stop = true;
                break;
            }
            guard += 1;
            if guard > 50_000_000 {
                self.error("Too many commands executed (endless loop?)");
                break;
            }
            let raw = prog[pc].trim().to_string();
            if raw.is_empty() || raw.starts_with(':') {
                pc += 1;
                continue;
            }
            self.lines_run += 1;
            let up = raw.to_uppercase();
            let head = up.split(',').next().unwrap_or("").trim().to_string();
            // ------------------------------------------------ control flow
            match head.as_str() {
                "*DO" => {
                    let f = split_fields(&self.subst(&raw));
                    let var = Self::txt(&f, 1);
                    let a = self.num(&f, 2, 0.);
                    let b = self.num(&f, 3, 0.);
                    let inc = self.num(&f, 4, 1.);
                    let inc = if inc == 0. { 1. } else { inc };
                    if (inc > 0. && a > b + 1e-12) || (inc < 0. && a < b - 1e-12) {
                        match Self::find_block_end(prog, pc + 1, &["*DO"], &["*ENDDO"], &[]) {
                            Some(e) => pc = e + 1,
                            None => {
                                self.error("*DO without *ENDDO");
                                return;
                            }
                        }
                        continue;
                    }
                    self.params.insert(var.clone(), Param::Num(a));
                    stack.push(Frame::Loop { var, end: b, inc, start: pc });
                    pc += 1;
                    continue;
                }
                "*ENDDO" => {
                    let top = stack.iter().rposition(|f| matches!(f, Frame::Loop { .. }));
                    if let Some(t) = top {
                        stack.truncate(t + 1);
                        if let Some(Frame::Loop { var, end, inc, start }) = stack.last() {
                            let (var, end, inc, start) = (var.clone(), *end, *inc, *start);
                            let v = match self.params.get(&var) {
                                Some(Param::Num(v)) => *v,
                                _ => end,
                            } + inc;
                            if (inc > 0. && v <= end + 1e-9) || (inc < 0. && v >= end - 1e-9) {
                                self.params.insert(var, Param::Num(v));
                                pc = start + 1;
                                continue;
                            }
                            stack.pop();
                        }
                    }
                    pc += 1;
                    continue;
                }
                "*CYCLE" | "*EXIT" => {
                    let exit = head == "*EXIT";
                    match Self::find_block_end(prog, pc + 1, &["*DO"], &["*ENDDO"], &[]) {
                        Some(e) => {
                            if exit {
                                if let Some(t) = stack.iter().rposition(|f| matches!(f, Frame::Loop { .. })) {
                                    stack.truncate(t);
                                }
                                pc = e + 1;
                            } else {
                                pc = e;
                            }
                        }
                        None => pc += 1,
                    }
                    continue;
                }
                "*IF" => {
                    let f = split_fields(&self.subst(&raw));
                    let base = Self::txt(&f, 4);
                    let c = self.cond(&f);
                    if base == "THEN" {
                        if c {
                            stack.push(Frame::If { taken: true });
                            pc += 1;
                        } else {
                            stack.push(Frame::If { taken: false });
                            match Self::find_block_end(prog, pc + 1, &["*IF"], &["*ENDIF"], &["*ELSEIF", "*ELSE"]) {
                                Some(e) => pc = e,
                                None => {
                                    self.error("*IF without *ENDIF");
                                    return;
                                }
                            }
                        }
                        continue;
                    }
                    if c {
                        match base.as_str() {
                            "EXIT" | "CYCLE" => {
                                if let Some(e) = Self::find_block_end(prog, pc + 1, &["*DO"], &["*ENDDO"], &[]) {
                                    if base == "EXIT" {
                                        if let Some(t) = stack.iter().rposition(|f| matches!(f, Frame::Loop { .. })) {
                                            stack.truncate(t);
                                        }
                                        pc = e + 1;
                                    } else {
                                        pc = e;
                                    }
                                    continue;
                                }
                            }
                            "STOP" => {
                                self.stop = true;
                                return;
                            }
                            b if b.starts_with(':') => {
                                if let Some(i) = prog.iter().position(|l| l.trim().to_uppercase() == b) {
                                    pc = i + 1;
                                    continue;
                                }
                                self.error(format!("label {b} not found"));
                            }
                            _ => {}
                        }
                    }
                    pc += 1;
                    continue;
                }
                "*ELSEIF" => {
                    let taken = matches!(stack.last(), Some(Frame::If { taken: true }));
                    if taken {
                        pc = Self::find_block_end(prog, pc + 1, &["*IF"], &["*ENDIF"], &[]).unwrap_or(prog.len());
                        continue;
                    }
                    let f = split_fields(&self.subst(&raw));
                    if self.cond(&f) {
                        if let Some(Frame::If { taken }) = stack.last_mut() {
                            *taken = true;
                        }
                        pc += 1;
                    } else {
                        pc = Self::find_block_end(prog, pc + 1, &["*IF"], &["*ENDIF"], &["*ELSEIF", "*ELSE"]).unwrap_or(prog.len());
                    }
                    continue;
                }
                "*ELSE" => {
                    let taken = matches!(stack.last(), Some(Frame::If { taken: true }));
                    if taken {
                        pc = Self::find_block_end(prog, pc + 1, &["*IF"], &["*ENDIF"], &[]).unwrap_or(prog.len());
                    } else {
                        if let Some(Frame::If { taken }) = stack.last_mut() {
                            *taken = true;
                        }
                        pc += 1;
                    }
                    continue;
                }
                "*ENDIF" => {
                    if let Some(t) = stack.iter().rposition(|f| matches!(f, Frame::If { .. })) {
                        stack.truncate(t);
                    }
                    pc += 1;
                    continue;
                }
                "*GO" => {
                    let f = split_fields(&raw);
                    let lab = Self::txt(&f, 1);
                    if let Some(i) = prog.iter().position(|l| l.trim().to_uppercase() == lab) {
                        pc = i + 1;
                    } else {
                        self.error(format!("label {lab} not found"));
                        pc += 1;
                    }
                    continue;
                }
                _ => {}
            }
            // ------------------------------------------------ data blocks
            if head.starts_with("NBLO") || head.starts_with("EBLO") || head.starts_with("CMBL") {
                pc = self.data_block(prog, pc, &head, &raw);
                continue;
            }
            if head == "*VWRITE" || head == "*VWRI" || head == "*MSG" {
                let f = split_fields(&self.subst(&raw));
                let mut vals = Vec::new();
                for i in 1..f.len() {
                    if f[i].is_empty() {
                        continue;
                    }
                    let fi = f[i].clone();
                    match self.params.get(&fi.to_uppercase()) {
                        Some(Param::Str(s)) => vals.push(s.clone()),
                        _ => {
                            if head == "*MSG" && i == 1 {
                                continue;
                            }
                            match self.eval(&fi) {
                                Ok(v) => vals.push(post::g(v).trim().to_string()),
                                Err(_) => vals.push(fi.trim_matches('\'').to_string()),
                            }
                        }
                    }
                }
                pc += 1;
                // the format line(s)
                let mut fmt = String::new();
                while pc < prog.len() {
                    let l = prog[pc].trim();
                    if l.starts_with('(') || head == "*MSG" && fmt.is_empty() {
                        fmt.push_str(l);
                        pc += 1;
                        if head == "*MSG" {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if head == "*MSG" {
                    self.print(format!(" {}", self.subst(&fmt)));
                } else {
                    self.print(format!(" {}", vals.join("  ")));
                }
                continue;
            }
            let line = self.subst(&raw);
            self.command(&line);
            self.commands_run += 1;
            pc += 1;
        }
    }

    fn data_block(&mut self, prog: &[String], pc: usize, head: &str, raw: &str) -> usize {
        let f = split_fields(raw);
        let mut i = pc + 1;
        let fmt_line = prog.get(i).map(|s| s.trim().to_string()).unwrap_or_default();
        let widths = if fmt_line.starts_with('(') {
            i += 1;
            fortran_widths(&fmt_line)
        } else {
            vec![]
        };
        let split = |line: &str| -> Vec<String> {
            if widths.is_empty() {
                line.split_whitespace().map(|s| s.to_string()).collect()
            } else {
                let v = fixed_fields(line, &widths);
                if v.iter().all(|x| x.is_empty()) {
                    line.split_whitespace().map(|s| s.to_string()).collect()
                } else {
                    v
                }
            }
        };
        if head.starts_with("NBLO") {
            let nint = widths.iter().filter(|w| w.0).count().max(1);
            let mut count = 0;
            while i < prog.len() {
                let l = &prog[i];
                let t = l.trim();
                i += 1;
                if t.starts_with("-1") || t.to_uppercase().starts_with("N,") {
                    break;
                }
                if t.is_empty() {
                    continue;
                }
                let v = split(l);
                let id = pnum(v.first().map(|s| s.as_str()).unwrap_or("0")) as usize;
                if id == 0 {
                    continue;
                }
                let gx = |k: usize| v.get(nint + k).map(|s| pnum(s)).unwrap_or(0.);
                self.model.set_node(id, [gx(0), gx(1), gx(2)]);
                if v.len() > nint + 3 && (gx(3) != 0. || gx(4) != 0. || gx(5) != 0.) {
                    self.warnings.push(format!("Node {id} has rotated nodal coordinates; they are ignored (global system used)."));
                }
                count += 1;
            }
            self.echo(format!(" NBLOCK: {count} nodes read"));
            return i;
        }
        if head.starts_with("EBLO") {
            let solid = Self::txt(&f, 2) == "SOLID";
            let mut count = 0;
            while i < prog.len() {
                let l = prog[i].clone();
                let t = l.trim();
                i += 1;
                if t.starts_with("-1") {
                    break;
                }
                if t.is_empty() {
                    continue;
                }
                let v: Vec<i64> = split(&l).iter().map(|s| pnum(s) as i64).collect();
                if solid {
                    if v.len() < 11 {
                        continue;
                    }
                    let nn = v[8].max(0) as usize;
                    let mut nodes: Vec<usize> = v[11..].iter().map(|&x| x.max(0) as usize).collect();
                    while nodes.len() < nn && i < prog.len() {
                        let more: Vec<usize> = split(&prog[i]).iter().map(|s| pnum(s).max(0.) as usize).collect();
                        i += 1;
                        nodes.extend(more);
                    }
                    nodes.truncate(nn);
                    let e = Element { mat: v[0] as usize, etype: v[1] as usize, real: v[2] as usize, sec: v[3] as usize, nodes };
                    self.model.set_elem(v[10] as usize, e);
                } else {
                    if v.len() < 6 {
                        continue;
                    }
                    let mut nodes: Vec<usize> = v[5..].iter().map(|&x| x.max(0) as usize).collect();
                    while nodes.last() == Some(&0) {
                        nodes.pop();
                    }
                    let e = Element { etype: v[1] as usize, real: v[2] as usize, sec: v[3] as usize, mat: self.model.cur_mat, nodes }; // field 5 = element coordinate system
                    self.model.set_elem(v[0] as usize, e);
                }
                count += 1;
            }
            self.echo(format!(" EBLOCK: {count} elements read"));
            return i;
        }
        // CMBLOCK,name,NODE|ELEM,count
        let name = Self::txt(&f, 1);
        let is_node = Self::txt(&f, 2).starts_with("NODE");
        let cnt = pnum(f.get(3).map(|s| s.as_str()).unwrap_or("0")) as usize;
        let mut vals: Vec<i64> = Vec::new();
        while i < prog.len() && vals.len() < cnt {
            let v: Vec<i64> = split(&prog[i]).iter().filter(|s| !s.is_empty()).map(|s| pnum(s) as i64).collect();
            i += 1;
            vals.extend(v);
        }
        let mut ids = Vec::new();
        let mut last = 0i64;
        for v in vals {
            if v < 0 {
                for k in last + 1..=(-v) {
                    ids.push(k as usize);
                }
            } else {
                ids.push(v as usize);
                last = v;
            }
        }
        self.echo(format!(" CMBLOCK: component {name} ({} {})", ids.len(), if is_node { "nodes" } else { "elements" }));
        self.model.comps.insert(name, (is_node, ids));
        i
    }

    fn nodes_from(&mut self, f: &[String], i: usize, nend_i: usize, ninc_i: usize) -> Vec<usize> {
        let t = Self::txt(f, i);
        if t == "ALL" || t.is_empty() && i < f.len() && false {
            return self.model.sel_node_ids();
        }
        if t == "P" {
            self.error("Picking (P) is not available in batch input.");
            return vec![];
        }
        if let Some((true, ids)) = self.model.comps.get(&t).cloned() {
            return ids.into_iter().filter(|&n| self.model.node(n).is_some()).collect();
        }
        let a = self.num(f, i, 0.) as i64;
        let b = self.num(f, nend_i, a as f64) as i64;
        let c = self.num(f, ninc_i, 1.).max(1.) as i64;
        let mut v = Vec::new();
        let mut k = a;
        while k <= b.max(a) && k > 0 {
            v.push(k as usize);
            k += c;
        }
        v
    }

    fn elems_from(&mut self, f: &[String], i: usize) -> Vec<usize> {
        let t = Self::txt(f, i);
        if t == "ALL" {
            return self.model.sel_elem_ids();
        }
        if let Some((false, ids)) = self.model.comps.get(&t).cloned() {
            return ids;
        }
        let a = self.num(f, i, 0.) as usize;
        if a > 0 {
            vec![a]
        } else {
            vec![]
        }
    }

    fn mid(&mut self, a: usize, b: usize) -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&n) = self.mids.get(&key) {
            if self.model.node(n).is_some() {
                return n;
            }
        }
        let (p, q) = (self.model.node(a).unwrap_or([0.; 3]), self.model.node(b).unwrap_or([0.; 3]));
        let id = self.model.max_node() + 1;
        self.model.set_node(id, [(p[0] + q[0]) / 2., (p[1] + q[1]) / 2., (p[2] + q[2]) / 2.]);
        self.mids.insert(key, id);
        id
    }

    /// fill missing mid-side nodes for quadratic elements given by corners only
    fn complete_midsides(&mut self, kind: Option<Kind>, mut n: Vec<usize>) -> Vec<usize> {
        use crate::elements::{HEX20_EDGES, QUAD8_EDGES, TET10_EDGES};
        match kind {
            Some(Kind::Solid186) if n.iter().filter(|&&x| x > 0).count() == 8 || (n.len() == 8) => {
                n.resize(8, 0);
                for e in HEX20_EDGES {
                    let m = if n[e[0]] == n[e[1]] { n[e[0]] } else { self.mid(n[e[0]], n[e[1]]) };
                    n.push(m);
                }
                n
            }
            Some(Kind::Solid187) if n.len() == 4 => {
                for e in TET10_EDGES {
                    let m = self.mid(n[e[0]], n[e[1]]);
                    n.push(m);
                }
                n
            }
            Some(Kind::Plane183) if n.len() == 4 => {
                for e in QUAD8_EDGES {
                    let m = if n[e[0]] == n[e[1]] { n[e[0]] } else { self.mid(n[e[0]], n[e[1]]) };
                    n.push(m);
                }
                n
            }
            _ => n,
        }
    }

    fn add_element(&mut self, id: usize, nodes: Vec<usize>, complete: bool) {
        let m = &self.model;
        if !m.etypes.contains_key(&m.cur_type) {
            self.error(format!("Element type {} is not defined (use ET first).", m.cur_type));
            return;
        }
        let kind = m.etypes[&m.cur_type].kind;
        let nodes = if complete { self.complete_midsides(kind, nodes) } else { nodes };
        let e = Element { etype: self.model.cur_type, mat: self.model.cur_mat, real: self.model.cur_real, sec: self.model.cur_sec, nodes };
        self.model.set_elem(id, e);
    }

    fn select(&mut self, is_node: bool, f: &[String]) {
        let typ = {
            let t = Self::txt(f, 1);
            if t.is_empty() {
                "S".to_string()
            } else {
                t
            }
        };
        let item = Self::txt(f, 2);
        let comp = Self::txt(f, 3);
        let len = if is_node { self.model.nodes.len() } else { self.model.elems.len() };
        let exists = |m: &Model, i: usize| if is_node { m.nodes[i].is_some() } else { m.elems[i].is_some() };
        match typ.as_str() {
            "ALL" | "NONE" | "INVE" => {
                for i in 0..len {
                    if !exists(&self.model, i) {
                        continue;
                    }
                    let s = if is_node { &mut self.model.nsel[i] } else { &mut self.model.esel[i] };
                    *s = match typ.as_str() {
                        "ALL" => true,
                        "NONE" => false,
                        _ => !*s,
                    };
                }
                return;
            }
            _ => {}
        }
        let vmin = self.num(f, 4, 0.);
        let vmax0 = self.num(f, 5, vmin);
        let vinc = self.num(f, 6, 1.).max(1.);
        let vmax = vmax0.max(vmin);
        // selection tolerance of the original (measured on it, tests/seltol/probe.dat):
        // range -> 1e-8 x (VMAX - VMIN); single value -> 0.005 x |VMIN|; value 0 -> 1e-6
        let tol = if vmax > vmin {
            SEL_TOL_RANGE * (vmax - vmin)
        } else if vmin != 0. {
            SEL_TOL_VALUE * vmin.abs()
        } else {
            SEL_TOL_ZERO
        };
        let mut hit = vec![false; len];
        let ename_kind = if item == "ENAME" { Kind::from_name(&f.get(4).cloned().unwrap_or_default()).map(|k| k.0) } else { None };
        for i in 0..len {
            if !exists(&self.model, i) {
                continue;
            }
            let m = &self.model;
            hit[i] = match item.as_str() {
                "NODE" | "ELEM" | "" => {
                    let x = i as f64;
                    x >= vmin - 1e-9 && x <= vmax + 1e-9 && ((x - vmin) / vinc).fract().abs() < 1e-9
                }
                "LOC" if is_node => {
                    let c = match comp.as_str() {
                        "Y" => 1,
                        "Z" => 2,
                        _ => 0,
                    };
                    let v = m.nodes[i].unwrap()[c];
                    v >= vmin - tol && v <= vmax + tol
                }
                "CENT" if !is_node => {
                    let e = m.elems[i].as_ref().unwrap();
                    let c = match comp.as_str() {
                        "Y" => 1,
                        "Z" => 2,
                        _ => 0,
                    };
                    let ns: Vec<f64> = e.nodes.iter().filter_map(|&n| m.node(n)).map(|x| x[c]).collect();
                    let v = ns.iter().sum::<f64>() / ns.len().max(1) as f64;
                    v >= vmin - tol && v <= vmax + tol
                }
                "TYPE" | "MAT" | "REAL" | "SEC" | "SECN" if !is_node => {
                    let e = m.elems[i].as_ref().unwrap();
                    let v = match item.as_str() {
                        "TYPE" => e.etype,
                        "MAT" => e.mat,
                        "REAL" => e.real,
                        _ => e.sec,
                    } as f64;
                    v >= vmin - 1e-9 && v <= vmax + 1e-9
                }
                "ENAME" if !is_node => {
                    let e = m.elems[i].as_ref().unwrap();
                    m.kind_of(e).is_some() && m.kind_of(e) == ename_kind
                }
                _ => false,
            };
        }
        if !matches!(item.as_str(), "NODE" | "ELEM" | "" | "LOC" | "CENT" | "TYPE" | "MAT" | "REAL" | "SEC" | "SECN" | "ENAME") {
            self.warn(format!("{}SEL item {item} is not supported; selection unchanged.", if is_node { "N" } else { "E" }));
            return;
        }
        self.apply_sel(is_node, &typ, &hit);
    }

    fn apply_sel(&mut self, is_node: bool, typ: &str, hit: &[bool]) {
        let len = hit.len();
        for i in 0..len {
            let ex = if is_node { self.model.nodes[i].is_some() } else { self.model.elems[i].is_some() };
            if !ex {
                continue;
            }
            let s = if is_node { &mut self.model.nsel[i] } else { &mut self.model.esel[i] };
            *s = match typ {
                "S" => hit[i],
                "R" => *s && hit[i],
                "A" => *s || hit[i],
                "U" => *s && !hit[i],
                _ => *s,
            };
        }
    }

    fn sel_count(&self) -> (usize, usize) {
        (self.model.sel_node_ids().len(), self.model.sel_elem_ids().len())
    }

    fn command(&mut self, line: &str) {
        let f = split_fields(line);
        let name = f[0].trim().to_uppercase();
        // parameter assignment
        if !name.starts_with('/') && !name.starts_with('*') {
            if let Some(eqp) = line.find('=') {
                let lhs = line[..eqp].trim();
                let is_ident = lhs.chars().next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false)
                    && !lhs.contains(',');
                if is_ident {
                    let rhs = line[eqp + 1..].trim().to_string();
                    self.assign(lhs, &rhs);
                    return;
                }
            }
        }
        let key: String = if name.starts_with('*') || name.starts_with('/') {
            name.chars().take(5).collect()
        } else {
            name.chars().take(4).collect()
        };
        let m = &mut self.model;
        let _ = m;
        match key.as_str() {
            // ---------------------------------------------------------------- processors & session
            "/PREP" => {
                self.processor = "PREP7".into();
                self.echo("\n ***** MAPDL-COMPATIBLE FAST SOLVER: PREPROCESSOR (PREP7) *****");
            }
            "/SOLU" => {
                self.processor = "SOLU".into();
                self.echo("\n *****  SOLUTION ROUTINE  *****");
            }
            "/POST" => {
                self.processor = "POST1".into();
                if self.cur.is_none() && !self.results.is_empty() {
                    self.cur = Some(self.results.len() - 1);
                }
                self.echo("\n *****  GENERAL POSTPROCESSOR (POST1)  *****");
            }
            "FINI" => {
                self.processor = "BEGIN".into();
                self.echo(" FINISH");
            }
            "/TITL" => {
                self.title = line.splitn(2, ',').nth(1).unwrap_or("").trim().to_string();
                self.echo(format!(" TITLE=\n {}", self.title));
            }
            "/STIT" => {}
            "/COM" | "/COM " => {
                let t = line.splitn(2, ',').nth(1).unwrap_or("");
                self.print(format!(" {t}"));
            }
            "/NOPR" => self.nopr = true,
            "/GOPR" | "/GO" => self.nopr = false,
            "/EOF" | "/EXIT" | "/EXI" => self.stop = true,
            "/CLEA" => {
                self.model = Model::new();
                self.results.clear();
                self.cur = None;
                self.mids.clear();
                self.echo(" DATABASE CLEARED");
            }
            "/FILN" => {
                let j = Self::txt(&f, 1);
                if !j.is_empty() {
                    self.jobname = j.to_lowercase();
                }
            }
            "/CWD" => {
                let p = f.get(1).cloned().unwrap_or_default().trim_matches('\'').to_string();
                let pb = PathBuf::from(&p);
                self.dir = if pb.is_absolute() { pb } else { self.dir.join(pb) };
            }
            "/INPU" => {
                let mut fname = f.get(1).cloned().unwrap_or_default().trim_matches('\'').to_string();
                let ext = f.get(2).cloned().unwrap_or_default().trim_matches('\'').to_string();
                let dir = f.get(3).cloned().unwrap_or_default().trim_matches('\'').to_string();
                if !ext.is_empty() {
                    fname = format!("{fname}.{ext}");
                }
                let mut p = PathBuf::from(&fname);
                if !dir.is_empty() {
                    p = PathBuf::from(dir).join(p);
                }
                if !p.is_absolute() {
                    p = self.dir.join(p);
                }
                self.echo(format!(" /INPUT FILE= {}", p.display()));
                if let Err(e) = self.run_file(&p) {
                    self.error(e);
                }
                self.stop = false;
            }
            "*AFUN" => self.degrees = Self::txt(&f, 1) == "DEG",
            "*SET" => {
                let n = f.get(1).cloned().unwrap_or_default();
                let v = f.get(2).cloned().unwrap_or_default();
                self.assign(&n, &v);
            }
            "*DIM" => {
                let n = Self::txt(&f, 1);
                let t = Self::txt(&f, 2);
                if t.starts_with("STRI") || t.starts_with("CHAR") {
                    self.params.insert(n, Param::Str(String::new()));
                } else {
                    let d = [self.num(&f, 3, 1.).max(1.) as usize, self.num(&f, 4, 1.).max(1.) as usize, self.num(&f, 5, 1.).max(1.) as usize];
                    self.params.insert(n, Param::Arr { d, v: vec![0.; d[0] * d[1] * d[2]] });
                }
            }
            "*GET" => self.get(&f),
            "*STAT" => {
                let mut s = String::from("\n PARAMETER STATUS\n");
                for (k, v) in &self.params {
                    let _ = match v {
                        Param::Num(x) => writeln!(s, " {:<32} {}", k, post::g(*x)),
                        Param::Str(x) => writeln!(s, " {:<32} '{}'  CHARACTER", k, x),
                        Param::Arr { d, .. } => writeln!(s, " {:<32} {}x{}x{}  ARRAY", k, d[0], d[1], d[2]),
                    };
                }
                self.print(s);
            }
            "*CFOP" => {
                let mut fname = f.get(1).cloned().unwrap_or_default().trim_matches('\'').to_string();
                let ext = f.get(2).cloned().unwrap_or_default();
                if !ext.is_empty() {
                    fname = format!("{fname}.{}", ext.trim_matches('\''));
                }
                let append = Self::txt(&f, 4) == "APPEND";
                let p = self.dir.join(fname);
                self.cf = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(p).ok();
            }
            "*CFWR" => {
                let t = line.splitn(2, ',').nth(1).unwrap_or("").to_string();
                if let Some(fh) = self.cf.as_mut() {
                    use std::io::Write;
                    let _ = writeln!(fh, "{t}");
                }
            }
            "*CFCL" => self.cf = None,
            "*USE" | "*CREA" | "*END" | "*VFIL" | "*VGET" | "*VPUT" | "*VOPE" | "*VSCF" | "*MOPE" | "*MWRI" => {
                self.unsupported(&name);
            }
            // ---------------------------------------------------------------- model definition
            "ET" => {
                let it = self.num(&f, 1, 0.) as usize;
                let en = f.get(2).cloned().unwrap_or_default();
                let (kind, nm) = match Kind::from_name(&en) {
                    Some((k, n)) => (Some(k), n.to_string()),
                    None => (None, en.trim().to_uppercase()),
                };
                let mut ko = [0i32; 19];
                for k in 0..6 {
                    ko[k + 1] = self.num(&f, 3 + k, 0.) as i32;
                }
                if kind.is_none() {
                    self.warn(format!(
                        "Element type {nm} is not supported in this version; its elements will be ignored during solution."
                    ));
                }
                if nm == "PLANE182" || nm == "SOLID185" {
                    // nothing extra
                }
                self.echo(format!(" ELEMENT TYPE {it} IS {nm}"));
                self.model.etypes.insert(it, EType { name: nm, kind, keyopt: ko });
            }
            "KEYO" => {
                let it = self.num(&f, 1, 0.) as usize;
                let k = self.num(&f, 2, 0.) as usize;
                let v = self.num(&f, 3, 0.) as i32;
                if let Some(t) = self.model.etypes.get_mut(&it) {
                    if k < 19 {
                        t.keyopt[k] = v;
                    }
                }
            }
            "TYPE" => self.model.cur_type = self.num(&f, 1, 1.) as usize,
            "MAT" => self.model.cur_mat = self.num(&f, 1, 1.) as usize,
            "REAL" => self.model.cur_real = self.num(&f, 1, 1.) as usize,
            "SECN" => self.model.cur_sec = self.num(&f, 1, 1.) as usize,
            "ESYS" => {
                if self.num(&f, 1, 0.) != 0. {
                    self.warn("ESYS (element coordinate systems) is ignored; isotropic materials do not need it.");
                }
            }
            "MP" => {
                let lab = Self::txt(&f, 1);
                let mat = self.num(&f, 2, 1.) as usize;
                let v = self.num(&f, 3, 0.);
                if self.num(&f, 4, 0.) != 0. {
                    self.warn("Temperature-dependent MP coefficients are ignored (constant term used).");
                }
                self.set_mp(&lab, mat, v);
            }
            "MPDA" => {
                let lab = Self::txt(&f, 1);
                let mat = self.num(&f, 2, 1.) as usize;
                let st = self.num(&f, 3, 1.).max(1.) as usize;
                if st == 1 {
                    let v = self.num(&f, 4, 0.);
                    self.set_mp(&lab, mat, v);
                }
                if f.len() > 5 && f[5..].iter().any(|s| !s.is_empty()) {
                    self.warnings.push("MPDATA with several temperatures: the first value is used.".into());
                }
            }
            "MPTE" | "MPAM" | "TREF" | "TOFF" => {}
            "TB" | "TBDA" | "TBTE" | "TBPT" | "TBFI" => {
                if key == "TB" {
                    self.warn("TB (nonlinear material) data is ignored: this version solves linear elastic materials.");
                }
            }
            "R" => {
                let set = self.num(&f, 1, 1.) as usize;
                let v: Vec<f64> = (2..f.len().max(2)).map(|i| self.num(&f, i, 0.)).collect();
                self.model.reals.insert(set, v);
            }
            "RMOR" => {
                let v: Vec<f64> = (1..f.len()).map(|i| self.num(&f, i, 0.)).collect();
                let k = self.model.reals.keys().last().copied().unwrap_or(1);
                self.model.reals.entry(k).or_default().extend(v);
            }
            "SECT" => {
                let id = self.num(&f, 1, 1.) as usize;
                let kind = Self::txt(&f, 2);
                let sub = Self::txt(&f, 3);
                // beam subtypes the original accepts (keyword table read by SECTYPE, 0x14654930b):
                // RECT QUAD CSOL CTUB CHAN Z L I T HATS HREC ASEC MESH
                const BEAM_SUBS: [&str; 13] = ["RECT", "QUAD", "CSOL", "CTUB", "CHAN", "Z", "L", "I", "T", "HATS", "HREC", "ASEC", "MESH"];
                if kind.starts_with("BEAM") && !sub.is_empty() && !BEAM_SUBS.iter().any(|b| sub.starts_with(b) && (b.len() >= 4 || sub == *b)) {
                    self.warn(format!("{sub} is not a valid beam section subtype name for SECID {id}. The SECTYPE command is ignored."));
                    return;
                }
                let kind = if kind.starts_with("BEAM") { "BEAM".to_string() } else if kind.starts_with("LINK") { "LINK".to_string() } else { kind };
                let sub = match sub.get(..4).unwrap_or(&sub) {
                    "CSOL" => "CSOLID".to_string(),
                    "CTUB" => "CTUBE".to_string(),
                    _ => sub.clone(),
                };
                self.model.sections.insert(id, Section { kind, subtype: sub, data: vec![] });
                self.model.cur_sec = id;
            }
            "SECD" => {
                let v: Vec<f64> = (1..f.len()).map(|i| self.num(&f, i, 0.)).collect();
                let id = self.model.cur_sec;
                let kind = self.model.sections.get(&id).map(|s| (s.kind.clone(), s.subtype.clone())).unwrap_or_default();
                // validation rules of the original SECDATA handler (0x14653d7b0)
                if kind.0 == "LINK" && v.first().copied().unwrap_or(0.) <= 0. {
                    self.warn(format!("Negative or zero area is not allowed for link section {id}. The SECDATA command is ignored."));
                    return;
                }
                if kind.0 == "BEAM" {
                    let area = match kind.1.as_str() {
                        "RECT" | "" => v.first().copied().unwrap_or(0.) * v.get(1).copied().unwrap_or(0.),
                        "CSOLID" => v.first().copied().unwrap_or(0.),
                        "CTUBE" => v.get(1).copied().unwrap_or(0.) - v.first().copied().unwrap_or(0.),
                        "ASEC" => v.first().copied().unwrap_or(0.),
                        _ => 1.,
                    };
                    if area <= 0. {
                        self.warn(format!("Beam section has zero or negative cross sectional area. Input data for section ID {id} is ignored."));
                        return;
                    }
                }
                let s = self.model.sections.entry(id).or_default();
                s.data = v;
            }
            "SECO" | "SECOF" | "SECCO" => {}
            "N" => {
                let mut id = self.num(&f, 1, 0.) as usize;
                if id == 0 {
                    id = self.model.max_node() + 1;
                }
                let x = [self.num(&f, 2, 0.), self.num(&f, 3, 0.), self.num(&f, 4, 0.)];
                if (5..8).any(|i| self.num(&f, i, 0.) != 0.) {
                    self.warnings.push(format!("Node {id}: nodal rotation angles are ignored."));
                }
                self.model.set_node(id, x);
            }
            "E" => {
                let nodes: Vec<usize> = (1..f.len()).map(|i| self.num(&f, i, 0.) as usize).collect();
                let mut nodes = nodes;
                while nodes.last() == Some(&0) {
                    nodes.pop();
                }
                let id = self.model.max_elem() + 1;
                self.add_element(id, nodes, false);
            }
            "EN" => {
                let id = self.num(&f, 1, 0.) as usize;
                let mut nodes: Vec<usize> = (2..f.len()).map(|i| self.num(&f, i, 0.) as usize).collect();
                while nodes.last() == Some(&0) {
                    nodes.pop();
                }
                let id = if id == 0 { self.model.max_elem() + 1 } else { id };
                self.add_element(id, nodes, false);
            }
            "EMOR" => {
                let more: Vec<usize> = (1..f.len()).map(|i| self.num(&f, i, 0.) as usize).collect();
                let id = self.model.max_elem();
                if let Some(Some(e)) = self.model.elems.get_mut(id) {
                    e.nodes.extend(more);
                    while e.nodes.last() == Some(&0) {
                        e.nodes.pop();
                    }
                }
            }
            "NGEN" => {
                let itime = self.num(&f, 1, 1.) as i64;
                let inc = self.num(&f, 2, 0.) as i64;
                let n1 = self.num(&f, 3, 0.) as usize;
                let n2 = self.num(&f, 4, n1 as f64) as usize;
                let ninc = self.num(&f, 5, 1.).max(1.) as usize;
                let d = [self.num(&f, 6, 0.), self.num(&f, 7, 0.), self.num(&f, 8, 0.)];
                for t in 1..itime {
                    let mut n = n1;
                    while n <= n2 {
                        if let Some(x) = self.model.node(n) {
                            let id = (n as i64 + inc * t) as usize;
                            self.model.set_node(id, [x[0] + d[0] * t as f64, x[1] + d[1] * t as f64, x[2] + d[2] * t as f64]);
                        }
                        n += ninc;
                    }
                }
            }
            "FILL" => {
                let n1 = self.num(&f, 1, 0.) as usize;
                let n2 = self.num(&f, 2, 0.) as usize;
                let (Some(a), Some(b)) = (self.model.node(n1), self.model.node(n2)) else {
                    self.error("FILL: end nodes not defined");
                    return;
                };
                let nfill_def = (n2 as i64 - n1 as i64).abs() - 1;
                let nfill = self.num(&f, 3, nfill_def as f64) as i64;
                let nstrt = self.num(&f, 4, (n1 + 1) as f64) as i64;
                let ninc = self.num(&f, 5, 1.) as i64;
                for k in 0..nfill.max(0) {
                    let t = (k + 1) as f64 / (nfill + 1) as f64;
                    let id = (nstrt + k * ninc) as usize;
                    self.model.set_node(id, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
                }
            }
            "EGEN" => {
                let itime = self.num(&f, 1, 1.) as i64;
                let ninc = self.num(&f, 2, 0.) as i64;
                let e1 = self.num(&f, 3, 0.) as usize;
                let e2 = self.num(&f, 4, e1 as f64) as usize;
                let einc = self.num(&f, 5, 1.).max(1.) as usize;
                let base = self.model.max_elem();
                let mut created = 0;
                for t in 1..itime {
                    let mut e = e1;
                    while e <= e2 {
                        if let Some(Some(el)) = self.model.elems.get(e).cloned() {
                            let mut ne = el.clone();
                            ne.nodes = el.nodes.iter().map(|&n| if n == 0 { 0 } else { (n as i64 + ninc * t) as usize }).collect();
                            created += 1;
                            self.model.set_elem(base + created, ne);
                        }
                        e += einc;
                    }
                }
            }
            "NDEL" => {
                let ids = self.nodes_from(&f, 1, 2, 3);
                for n in ids {
                    if n < self.model.nodes.len() {
                        self.model.nodes[n] = None;
                    }
                    let before = self.model.ces.len();
                    self.model.ces.retain(|c| c.slave != n && c.master != n);
                    let k = before - self.model.ces.len();
                    if k > 0 {
                        self.warn(format!("Deleted node {n} was referenced in {k} constraint equations. These equations were also deleted."));
                    }
                }
            }
            "EDEL" => {
                let ids = self.elems_from(&f, 1);
                for e in ids {
                    if e < self.model.elems.len() {
                        self.model.elems[e] = None;
                    }
                }
            }
            "NUMM" => self.nummrg(),
            "NUMC" | "NUMS" | "NUMO" => {}
            "BLOC" | "RECT" => {
                let v = [self.num(&f, 1, 0.), self.num(&f, 2, 0.), self.num(&f, 3, 0.), self.num(&f, 4, 0.), self.num(&f, 5, 0.), self.num(&f, 6, 0.)];
                let is_area = key == "RECT";
                let vol = Volume {
                    min: [v[0].min(v[1]), v[2].min(v[3]), if is_area { 0. } else { v[4].min(v[5]) }],
                    max: [v[0].max(v[1]), v[2].max(v[3]), if is_area { 0. } else { v[4].max(v[5]) }],
                    is_area,
                    meshed: false,
                };
                self.model.volumes.push(vol);
                self.echo(format!(" CREATE A {} WITH {} = {}", if is_area { "RECTANGULAR AREA" } else { "HEXAHEDRAL VOLUME" }, if is_area { "AREA" } else { "VOLUME" }, self.model.volumes.len()));
            }
            "ESIZ" => {
                self.model.esize = self.num(&f, 1, 0.);
                self.model.ndiv = self.num(&f, 2, 0.) as usize;
            }
            "LESI" => {
                let size = self.num(&f, 2, 0.);
                let nd = self.num(&f, 4, 0.) as usize;
                if Self::txt(&f, 1) != "ALL" {
                    self.warn("LESIZE on individual lines is approximated: the setting is applied to all lines.");
                }
                if nd > 0 {
                    self.model.ndiv = nd;
                } else if size > 0. {
                    self.model.esize = size;
                }
            }
            "VMES" | "AMES" => self.mesh(key == "AMES"),
            "MSHK" | "MSHA" | "MOPT" | "SMRT" | "MSHM" | "ETCO" => {}
            "CSYS" | "DSYS" | "RSYS" => {
                if self.num(&f, 1, 0.) != 0. {
                    self.warn(format!("{name}: only the global Cartesian system (0) is supported; ignored."));
                }
            }
            "LOCA" | "NROT" | "CLOC" => self.warn(format!("{name}: local coordinate systems are not supported; ignored.")),
            // ---------------------------------------------------------------- selection
            "NSEL" => {
                self.select(true, &f);
            }
            "ESEL" => {
                self.select(false, &f);
            }
            "ALLS" => {
                for i in 0..self.model.nodes.len() {
                    self.model.nsel[i] = self.model.nodes[i].is_some();
                }
                for i in 0..self.model.elems.len() {
                    self.model.esel[i] = self.model.elems[i].is_some();
                }
            }
            "NSLE" => {
                let typ = if Self::txt(&f, 1).is_empty() { "S".to_string() } else { Self::txt(&f, 1) };
                let mut hit = vec![false; self.model.nodes.len()];
                for e in self.model.sel_elem_ids() {
                    for &n in &self.model.elems[e].as_ref().unwrap().nodes {
                        if n < hit.len() {
                            hit[n] = true;
                        }
                    }
                }
                self.apply_sel(true, &typ, &hit);
            }
            "ESLN" => {
                let typ = if Self::txt(&f, 1).is_empty() { "S".to_string() } else { Self::txt(&f, 1) };
                let all = self.num(&f, 2, 0.) != 0.;
                let mut hit = vec![false; self.model.elems.len()];
                for e in self.model.elem_ids().collect::<Vec<_>>() {
                    let el = self.model.elems[e].as_ref().unwrap();
                    let sel = |n: &usize| self.model.nsel.get(*n).copied().unwrap_or(false);
                    hit[e] = if all { el.nodes.iter().filter(|&&n| n > 0).all(sel) } else { el.nodes.iter().any(sel) };
                }
                self.apply_sel(false, &typ, &hit);
            }
            "CM" => {
                let nm = Self::txt(&f, 1);
                let ent = Self::txt(&f, 2);
                let is_node = ent.starts_with("NODE");
                let ids = if is_node { self.model.sel_node_ids() } else { self.model.sel_elem_ids() };
                self.model.comps.insert(nm, (is_node, ids));
            }
            "CMSE" => {
                let typ = if Self::txt(&f, 1).is_empty() { "S".to_string() } else { Self::txt(&f, 1) };
                if typ == "ALL" || typ == "NONE" {
                    return;
                }
                let nm = Self::txt(&f, 2);
                let Some((is_node, ids)) = self.model.comps.get(&nm).cloned() else {
                    self.warn(format!("Component {nm} is not defined."));
                    return;
                };
                let len = if is_node { self.model.nodes.len() } else { self.model.elems.len() };
                let mut hit = vec![false; len];
                for i in ids {
                    if i < len {
                        hit[i] = true;
                    }
                }
                self.apply_sel(is_node, &typ, &hit);
            }
            "CMDE" => {
                let nm = Self::txt(&f, 1);
                self.model.comps.remove(&nm);
            }
            "CMLI" => {
                let s: Vec<String> = self.model.comps.iter().map(|(k, v)| format!(" {k:<32} {} {}", if v.0 { "NODE" } else { "ELEM" }, v.1.len())).collect();
                self.print(format!("\n COMPONENTS\n{}", s.join("\n")));
            }
            // ---------------------------------------------------------------- loads
            "D" => {
                let lab = Self::txt(&f, 2);
                let v = self.num(&f, 3, 0.);
                if f.get(4).map(|s| !s.is_empty()).unwrap_or(false) && self.num(&f, 4, 0.) != 0. {
                    self.warn("D: imaginary value (VALUE2) is ignored in static analysis.");
                }
                let ids = self.nodes_from(&f, 1, 5, 6);
                let mut labs = vec![lab.clone()];
                for i in 7..12 {
                    let l = Self::txt(&f, i);
                    if !l.is_empty() {
                        labs.push(l);
                    }
                }
                for l in labs {
                    let dofs: Vec<u8> = if l == "ALL" {
                        (0..6).collect()
                    } else if let Some(d) = dof_of(&l) {
                        vec![d]
                    } else {
                        self.warn(format!("D label {l} is not supported (structural UX..ROTZ only)."));
                        continue;
                    };
                    for &n in &ids {
                        for &d in &dofs {
                            self.model.disp.insert((n, d), v);
                        }
                    }
                }
            }
            "DDEL" => {
                let lab = Self::txt(&f, 2);
                let ids = self.nodes_from(&f, 1, 3, 4);
                for n in ids {
                    for d in 0..6u8 {
                        if lab == "ALL" || lab.is_empty() || dof_of(&lab) == Some(d) {
                            self.model.disp.remove(&(n, d));
                        }
                    }
                }
            }
            "F" => {
                let lab = Self::txt(&f, 2);
                let v = self.num(&f, 3, 0.);
                let ids = self.nodes_from(&f, 1, 5, 6);
                match force_of(&lab) {
                    Some(d) => {
                        for n in ids {
                            self.model.force.insert((n, d), v);
                        }
                    }
                    None => self.warn(format!("F label {lab} is not supported (FX..MZ only).")),
                }
            }
            "FDEL" => {
                let lab = Self::txt(&f, 2);
                let ids = self.nodes_from(&f, 1, 3, 4);
                for n in ids {
                    for d in 0..6u8 {
                        if lab == "ALL" || lab.is_empty() || force_of(&lab) == Some(d) {
                            self.model.force.remove(&(n, d));
                        }
                    }
                }
            }
            "SF" => {
                let lab = Self::txt(&f, 2);
                if lab != "PRES" {
                    self.warn(format!("SF label {lab} is not supported (PRES only)."));
                    return;
                }
                let v = self.num(&f, 3, 0.);
                let ids = self.nodes_from(&f, 1, 99, 99);
                for n in ids {
                    self.model.node_pres.insert(n, v);
                }
            }
            "SFDE" => {
                let ids = self.nodes_from(&f, 1, 99, 99);
                for n in ids {
                    self.model.node_pres.remove(&n);
                }
            }
            "SFE" => {
                let lkey = self.num(&f, 2, 1.) as usize;
                let lab = Self::txt(&f, 3);
                if lab != "PRES" {
                    self.warn(format!("SFE label {lab} is not supported (PRES only)."));
                    return;
                }
                let kval = self.num(&f, 4, 0.);
                if kval != 0. {
                    return;
                }
                let v = self.num(&f, 5, 0.);
                for e in self.elems_from(&f, 1) {
                    self.model.elem_pres.insert((e, lkey), v);
                }
            }
            "SFED" => {
                for e in self.elems_from(&f, 1) {
                    self.model.elem_pres.retain(|k, _| k.0 != e);
                }
            }
            "ACEL" => self.model.acel = [self.num(&f, 1, 0.), self.num(&f, 2, 0.), self.num(&f, 3, 0.)],
            "BF" | "BFE" | "BFUN" | "TUNI" => self.warn(format!("{name}: thermal body loads are not supported in this version; ignored.")),
            // ---------------------------------------------------------------- solution
            "ANTY" => {
                let t = Self::txt(&f, 1);
                if !(t.is_empty() || t == "0" || t.starts_with("STAT")) {
                    self.error(format!("ANTYPE,{t}: only static analysis is available in this version."));
                }
            }
            "NLGE" => {
                if Self::txt(&f, 1) == "ON" {
                    self.warn("NLGEOM,ON is not supported: the analysis is solved as linear (small deflection).");
                }
            }
            "EQSL" => {
                let t = Self::txt(&f, 1);
                self.settings.solver = if t.starts_with("PCG") || t.starts_with("JCG") || t.starts_with("ICCG") { SolverKind::Pcg } else { SolverKind::Sparse };
                // EQSLVE (0x14e4f4430): TOLER >= 1.0 gives a warning; TOLER > 0 is used, otherwise the
                // default 1.0E-8 stored at 0x156426430
                let tol = self.num(&f, 2, 0.);
                if self.settings.solver == SolverKind::Pcg && tol >= 1.0 {
                    self.warn(format!("The tolerance value input for the PCG equation solver ({tol}) is greater than or equal to 1.0."));
                }
                self.settings.pcg_tol = if tol > 0. { tol } else { crate::solver::PCG_DEFAULT_TOL };
                self.echo(format!(" EQUATION SOLVER OPTION = {}", if self.settings.solver == SolverKind::Pcg { "PCG" } else { "SPARSE" }));
            }
            "CE" => {
                // CE,NEQN,CONST,NODE1,Lab1,C1,NODE2,Lab2,C2,... (rules of ceput, FUN_142dacf50)
                let nlab = Self::txt(&f, 1);
                let highest = self.model.ces.iter().map(|c| c.n).max().unwrap_or(0);
                let neqn = if nlab.is_empty() || nlab == "NEXT" || nlab == "HIGH" { highest + 1 } else { self.num(&f, 1, 0.).max(1.) as usize };
                let cst = self.num(&f, 2, 0.);
                // terms; duplicates (same node and DOF) are merged first (REMOVEDUPTERMS)
                let mut terms: Vec<(usize, u8, f64)> = Vec::new();
                let mut bad = false;
                let mut i = 3;
                while i + 2 < f.len().max(i + 3) && i < f.len() {
                    let node = self.num(&f, i, 0.) as usize;
                    let lab = Self::txt(&f, i + 1);
                    let co = self.num(&f, i + 2, 1.);
                    if node > 0 {
                        match dof_of(&lab) {
                            Some(d) => match terms.iter_mut().find(|t| t.0 == node && t.1 == d) {
                                Some(t) => t.2 += co,
                                None => terms.push((node, d, co)),
                            },
                            None => bad = true,
                        }
                    }
                    i += 3;
                }
                self.model.ces.retain(|c| c.n != neqn); // CE with an existing number replaces it
                // undefined nodes are refused at input (original run: tests/cerules/ce_rules_orig.out)
                let undefined: Vec<usize> = terms.iter().map(|t| t.0).filter(|&n| self.model.node(n).is_none()).collect();
                if let Some(n) = undefined.first() {
                    self.warn(format!("Node {n} is undefined.  Constraint equation {neqn} cannot be generated.  The CE command is ignored."));
                    return;
                }
                if !bad && !(terms.len() == 1 && terms[0].2 == 0.) {
                    self.echo(format!(" CONSTRAINT EQUATION NO. {:>10} HAS {:>8} TERMS.  CONSTANT= {}", neqn, terms.len(), post::g(cst).trim()));
                }
                if bad {
                    self.warn("CE: only structural DOF labels (UX, UY, UZ, ROTX, ROTY, ROTZ) are supported; equation ignored.");
                } else if terms.len() == 1 && terms[0].2 == 0. {
                    self.warn(format!("Constraint equation {neqn} is a trivial equation (such as 5 UX = 5 UX) and will be ignored."));
                } else if terms.len() == 1 {
                    let (n1, d1, c1) = terms[0];
                    self.model.ces.push(Ce { n: neqn, slave: n1, master: 0, dof: d1, c: cst / c1 });
                } else if terms.len() == 2 && terms[0].1 == terms[1].1 && terms[0].2 != 0. && (terms[0].2 + terms[1].2).abs() <= 1e-12 * terms[0].2.abs() {
                    // c1 u1 - c1 u2 = CONST  ->  u1 - u2 = CONST / c1; the first term is the dependent one
                    let (n1, d1, c1) = terms[0];
                    self.model.ces.push(Ce { n: neqn, slave: n1, master: terms[1].0, dof: d1, c: cst / c1 });
                } else {
                    self.warn(format!(
                        "CE {neqn}: this version supports one-term equations and two-term equations c*u1 - c*u2 = CONST on the same DOF; ignored."
                    ));
                }
            }
            "CEDE" => {
                // CEDELE,NE1,NE2,NINC / ALL ; a missing number: "Constraint equation %I does not exist." (0x155dc6380)
                if Self::txt(&f, 1) == "ALL" || f.len() < 2 || f[1].trim().is_empty() {
                    self.model.ces.clear();
                } else {
                    let a = self.num(&f, 1, 0.) as usize;
                    let b = self.num(&f, 2, a as f64) as usize;
                    let inc = (self.num(&f, 3, 1.) as usize).max(1);
                    let mut k = a;
                    while k <= b.max(a) {
                        if self.model.ces.iter().any(|c| c.n == k) {
                            self.model.ces.retain(|c| c.n != k);
                        } else {
                            self.warn(format!("Entity {k} is undefined.  The CEDELE command is ignored."));
                        }
                        k += inc;
                    }
                }
            }
            "ERES" => {
                // ERESX (0x14d03e230): blank or DEFA -> default (extrapolate for linear materials);
                // YES -> extrapolate; NO -> copy integration-point results to the nodes
                let k = Self::txt(&f, 1);
                self.model.eresx_copy = k.starts_with("NO");
            }
            "SOLV" => self.solve(),
            "OUTR" | "OUTP" | "TIME" | "NSUB" | "AUTO" | "KBC" | "DELT" | "RESC" | "CNVT" | "NROP" | "LUMP" | "PSTR" | "SSTI" | "EMAT" | "LNSR" | "PRED" | "NEQI" | "BCSO" | "DMPO" | "PCGO" | "RATE" | "/STAT" | "/CONF" | "SAVE" | "RESU" | "/RUNS" => {}
            // ---------------------------------------------------------------- post
            "SET" | "SUBS" => {
                if self.results.is_empty() {
                    self.warn("No results are available (SOLVE first).");
                } else {
                    let ls = Self::txt(&f, 1);
                    let idx = if ls == "LAST" || ls.is_empty() { self.results.len() - 1 } else if ls == "FIRS" || ls == "FIRST" { 0 } else { (self.num(&f, 1, 1.) as usize).max(1) - 1 };
                    self.cur = Some(idx.min(self.results.len() - 1));
                }
            }
            "PRNS" => {
                let (it, c) = (Self::txt(&f, 1), Self::txt(&f, 2));
                let it = if it.is_empty() { "U".to_string() } else { it };
                match self.result() {
                    Some(r) => {
                        let s = post::prnsol(&self.model, r, &it, &c);
                        self.print(s);
                    }
                    None => self.warn("PRNSOL: no results (SOLVE first)."),
                }
            }
            "PRRS" | "PRRF" => match self.result() {
                Some(r) => {
                    let s = post::prrsol(&self.model, r);
                    self.print(s);
                }
                None => self.warn("PRRSOL: no results."),
            },
            "PRES" | "PRET" => match self.result() {
                Some(r) => {
                    let s = post::presol(&self.model, r);
                    self.print(s);
                }
                None => self.warn("PRESOL: no results."),
            },
            "FSUM" => match self.result() {
                Some(r) => {
                    let (t, s) = post::fsum(&self.model, r);
                    self.last_fsum = t;
                    self.print(s);
                }
                None => self.warn("FSUM: no results."),
            },
            "PLNS" | "NSOR" => {
                let (it, c) = (Self::txt(&f, 1), Self::txt(&f, 2));
                let it = if it.is_empty() { "U".to_string() } else { it };
                let c = if c.is_empty() && it == "U" { "SUM".to_string() } else if c.is_empty() { "EQV".to_string() } else { c };
                if let Some(r) = self.result() {
                    let mut mx = (f64::NEG_INFINITY, 0usize);
                    let mut mn = (f64::INFINITY, 0usize);
                    for n in self.model.sel_node_ids() {
                        if let Some(v) = post::node_value(&self.model, r, n, &it, &c) {
                            if v > mx.0 {
                                mx = (v, n);
                            }
                            if v < mn.0 {
                                mn = (v, n);
                            }
                        }
                    }
                    self.last_plnsol = (mx.0, mn.0, mx.1, mn.1);
                }
            }
            // ---------------------------------------------------------------- silently ignored (graphics, files)
            k if k.starts_with('/')
                && matches!(
                    k,
                    "/SHOW" | "/VIEW" | "/PNUM" | "/PBC" | "/REPL" | "/ANGL" | "/AUTO" | "/DEVI" | "/GRAP" | "/TRIA" | "/PSF" | "/PLOP" | "/CONT" | "/EDGE" | "/ESHA" | "/RGB" | "/TYPE" | "/DSCA" | "/FOCU" | "/NERR" | "/BATC" | "/UNIT" | "/OUTP" | "/WB" | "/GST" | "/NOLI" | "/INQU" | "/MKDI" | "/DELE" | "/RMDI" | "/ERAS" | "/WIND" | "/PLOP" | "/GLIN" | "/GFIL" | "/IMAG" | "/SYS" | "/PAGE" | "/HEAD" | "/FORM" | "/UIS" | "/MENU" | "/VUP" | "/ZOOM" | "/DIST" | "/USER" | "/XRAN" | "/YRAN" | "/AXLA" | "/GROP" | "/COLO" | "/NUMB" | "/DIVI" | "/RATI" | "/DSCALE" | "/PREC" | "/RESE" | "/DV3D" | "/ESHAPE"
                ) => {}
            "EPLO" | "NPLO" | "PLDI" | "PLES" | "GPLO" | "APLO" | "VPLO" | "LPLO" | "KPLO" | "NLIS" | "ELIS" | "DLIS" | "FLIS" | "ETLI" | "MPLI" | "*LIS" | "PLVE" | "ANDA" | "ANIM" => {}
            _ => self.unsupported(&name),
        }
    }

    fn unsupported(&mut self, name: &str) {
        let c = self.unsupported.entry(name.to_string()).or_insert(0);
        *c += 1;
        if *c == 1 {
            let _ = writeln!(self.out, "\n *** NOTE ***  Command {name} is not supported by this version and was skipped.");
        }
    }

    fn set_mp(&mut self, lab: &str, mat: usize, v: f64) {
        let lab = if lab == "NUXY" { "PRXY".to_string() } else { lab.to_string() };
        if matches!(lab.as_str(), "EY" | "EZ") {
            if self.model.mat_prop(mat, "EX").map(|e| (e - v).abs() > 1e-9 * e.abs()).unwrap_or(false) {
                self.warn(format!("Material {mat}: orthotropic properties are not supported; EX is used as isotropic modulus."));
            }
        }
        self.model.mats.entry(mat).or_default().insert(lab, v);
    }

    fn assign(&mut self, lhs: &str, rhs: &str) {
        let lhs = lhs.trim().to_uppercase();
        let rhs = rhs.trim();
        if let Some(p) = lhs.find('(') {
            let nm = lhs[..p].trim().to_string();
            let idx_s = lhs[p + 1..].trim_end_matches(')').to_string();
            let idx: Vec<usize> = split_fields(&idx_s).iter().map(|s| self.eval(s).unwrap_or(1.) as usize).collect();
            let vals: Vec<f64> = split_fields(rhs).iter().map(|s| self.eval(s).unwrap_or(0.)).collect();
            if let Some(Param::Arr { d, v }) = self.params.get_mut(&nm) {
                let i = idx.first().copied().unwrap_or(1).max(1);
                let j = idx.get(1).copied().unwrap_or(1).max(1);
                let k = idx.get(2).copied().unwrap_or(1).max(1);
                for (o, val) in vals.iter().enumerate() {
                    let pos = (i - 1 + o) + (j - 1) * d[0] + (k - 1) * d[0] * d[1];
                    if pos < v.len() {
                        v[pos] = *val;
                    }
                }
            } else {
                self.error(format!("Array {nm} is not dimensioned (*DIM)."));
            }
            return;
        }
        if rhs.is_empty() {
            self.params.remove(&lhs);
            return;
        }
        if rhs.starts_with('\'') {
            self.params.insert(lhs, Param::Str(rhs.trim_matches('\'').to_string()));
            return;
        }
        match self.eval(rhs) {
            Ok(v) => {
                self.params.insert(lhs, Param::Num(v));
            }
            Err(e) => {
                if let Some(Param::Str(s)) = self.params.get(&rhs.to_uppercase()).cloned() {
                    self.params.insert(lhs, Param::Str(s));
                } else {
                    self.error(format!("Cannot evaluate '{rhs}' for parameter {lhs}: {e}"));
                }
            }
        }
    }

    fn get(&mut self, f: &[String]) {
        let par = Self::txt(f, 1);
        let ent = Self::txt(f, 2);
        let num = self.num(f, 3, 0.) as usize;
        let i1 = Self::txt(f, 4);
        let i1n = Self::txt(f, 5);
        let i2 = Self::txt(f, 6);
        let idx5 = f.get(5).and_then(|s| self.eval(s).ok()).unwrap_or(1.).max(1.) as usize;
        let m = &self.model;
        let r = self.cur.and_then(|c| self.results.get(c));
        let v: Option<f64> = match ent.as_str() {
            "NODE" if num == 0 => match (i1.as_str(), i1n.as_str()) {
                ("COUNT", _) => Some(m.sel_node_ids().len() as f64),
                ("NUM", "MAX") => m.sel_node_ids().last().map(|&x| x as f64).or(Some(0.)),
                ("NUM", "MIN") => m.sel_node_ids().first().map(|&x| x as f64).or(Some(0.)),
                ("NUM", "MAXD") => Some(m.max_node() as f64),
                ("NUM", "MIND") => m.node_ids().next().map(|x| x as f64),
                ("MXLOC", c) | ("MNLOC", c) => {
                    let k = match c {
                        "Y" => 1,
                        "Z" => 2,
                        _ => 0,
                    };
                    let it = m.sel_node_ids().into_iter().filter_map(|n| m.node(n)).map(|x| x[k]);
                    if i1 == "MXLOC" {
                        it.fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))
                    } else {
                        it.fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.min(b))))
                    }
                }
                _ => None,
            },
            "NODE" => match i1.as_str() {
                "LOC" => post::node_value(m, r.unwrap_or(&ResultSet::default()), num, "LOC", &i1n),
                "U" | "ROT" | "S" => r.and_then(|r| post::node_value(m, r, num, &i1, if i1n.is_empty() { &i2 } else { &i1n })),
                "RF" => r.and_then(|r| post::node_value(m, r, num, "RF", &i1n)),
                "NSEL" => Some(if m.nsel.get(num).copied().unwrap_or(false) { 1. } else if m.node(num).is_some() { -1. } else { 0. }),
                _ => None,
            },
            "ELEM" if num == 0 => match (i1.as_str(), i1n.as_str()) {
                ("COUNT", _) => Some(m.sel_elem_ids().len() as f64),
                ("NUM", "MAX") => m.sel_elem_ids().last().map(|&x| x as f64).or(Some(0.)),
                ("NUM", "MIN") => m.sel_elem_ids().first().map(|&x| x as f64).or(Some(0.)),
                ("NUM", "MAXD") => Some(m.max_elem() as f64),
                _ => None,
            },
            "ELEM" => match i1.as_str() {
                "NODE" => m.elems.get(num).and_then(|e| e.as_ref()).and_then(|e| e.nodes.get(idx5 - 1)).map(|&n| n as f64),
                "ATTR" => m.elems.get(num).and_then(|e| e.as_ref()).map(|e| match i1n.as_str() {
                    "TYPE" => e.etype as f64,
                    "MAT" => e.mat as f64,
                    "REAL" => e.real as f64,
                    "SECN" => e.sec as f64,
                    _ => 0.,
                }),
                "SMIS" | "FORC" => r.and_then(|r| r.elem_forces.get(&num)).and_then(|v| v.get(idx5 - 1).copied()),
                _ => None,
            },
            "PLNS" | "PLNSOL" | "SORT" => match i1.as_str() {
                "MAX" => Some(self.last_plnsol.0),
                "MIN" => Some(self.last_plnsol.1),
                "IMAX" => Some(self.last_plnsol.2 as f64),
                "IMIN" => Some(self.last_plnsol.3 as f64),
                _ => None,
            },
            "FSUM" => force_of(&i1n).map(|d| self.last_fsum[d as usize]).or_else(|| force_of(&i1).map(|d| self.last_fsum[d as usize])),
            "ACTI" | "ACTIVE" => match i1.as_str() {
                "NUMCPU" => Some(self.settings.threads as f64),
                "TIME" => Some(0.),
                _ => None,
            },
            "COMMON" | "PARM" => None,
            _ => None,
        };
        match v {
            Some(v) => {
                self.params.insert(par, Param::Num(v));
            }
            None => {
                if ent == "ACTI" && i1 == "JOBNAM" {
                    let j = self.jobname.clone();
                    self.params.insert(par, Param::Str(j));
                } else {
                    self.warn(format!("*GET,{par},{ent},{num},{i1},{i1n}: this item is not available; parameter set to 0."));
                    self.params.insert(par, Param::Num(0.));
                }
            }
        }
    }

    fn nummrg(&mut self) {
        let (lo, hi) = self.model.bbox();
        let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1e-12);
        let tol = 1e-4 * diag;
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for n in self.model.node_ids().collect::<Vec<_>>() {
            let x = self.model.node(n).unwrap();
            let c = ((x[0] / tol).floor() as i64, (x[1] / tol).floor() as i64, (x[2] / tol).floor() as i64);
            let mut found = None;
            'o: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                            for &k in v {
                                let y = self.model.node(k).unwrap();
                                if (x[0] - y[0]).abs() <= tol && (x[1] - y[1]).abs() <= tol && (x[2] - y[2]).abs() <= tol {
                                    found = Some(k);
                                    break 'o;
                                }
                            }
                        }
                    }
                }
            }
            match found {
                Some(k) => {
                    remap.insert(n, k);
                }
                None => grid.entry(c).or_default().push(n),
            }
        }
        if remap.is_empty() {
            self.echo(" NUMMRG: no coincident nodes");
            return;
        }
        for e in self.model.elems.iter_mut().flatten() {
            for n in e.nodes.iter_mut() {
                if let Some(&k) = remap.get(n) {
                    *n = k;
                }
            }
        }
        let remap_map = |m: &mut BTreeMap<(usize, u8), f64>| {
            let old = std::mem::take(m);
            for ((n, d), v) in old {
                m.insert((*remap.get(&n).unwrap_or(&n), d), v);
            }
        };
        remap_map(&mut self.model.disp);
        remap_map(&mut self.model.force);
        for (&n, _) in &remap {
            self.model.nodes[n] = None;
            self.model.nsel[n] = false;
        }
        self.mids.clear();
        self.echo(format!(" NUMMRG: {} coincident nodes merged", remap.len()));
    }

    fn mesh(&mut self, areas: bool) {
        let kind = self.model.etypes.get(&self.model.cur_type).and_then(|t| t.kind);
        let ok = if areas { matches!(kind, Some(Kind::Plane182) | Some(Kind::Plane183)) } else { matches!(kind, Some(k) if k.is_solid()) };
        if !ok {
            self.error(format!(
                "{}MESH: the current element type (TYPE,{}) must be a {} element.",
                if areas { "A" } else { "V" },
                self.model.cur_type,
                if areas { "PLANE182/183" } else { "SOLID185/186/187" }
            ));
            return;
        }
        let kind = kind.unwrap();
        let vols: Vec<usize> = (0..self.model.volumes.len()).filter(|&i| self.model.volumes[i].is_area == areas && !self.model.volumes[i].meshed).collect();
        if vols.is_empty() {
            self.warn("Nothing to mesh (define BLOCK or RECTNG first).");
            return;
        }
        let mut ne = 0;
        let mut nn0 = self.model.node_count();
        for vi in vols {
            let v = self.model.volumes[vi].clone();
            let dims = [v.max[0] - v.min[0], v.max[1] - v.min[1], v.max[2] - v.min[2]];
            let div = |len: f64, m: &Model| -> usize {
                if len <= 0. {
                    0
                } else if m.ndiv > 0 {
                    m.ndiv
                } else if m.esize > 0. {
                    ((len / m.esize).round() as usize).max(1)
                } else {
                    4
                }
            };
            let nx = div(dims[0], &self.model).max(1);
            let ny = div(dims[1], &self.model).max(1);
            let nz = if areas { 0 } else { div(dims[2], &self.model).max(1) };
            let base = self.model.max_node() + 1;
            let idx = |i: usize, j: usize, k: usize| base + i + (nx + 1) * (j + (ny + 1) * k);
            for k in 0..=nz {
                for j in 0..=ny {
                    for i in 0..=nx {
                        let x = [
                            v.min[0] + dims[0] * i as f64 / nx as f64,
                            v.min[1] + dims[1] * j as f64 / ny as f64,
                            if areas { 0. } else { v.min[2] + dims[2] * k as f64 / nz as f64 },
                        ];
                        self.model.set_node(idx(i, j, k), x);
                    }
                }
            }
            let mut eid = self.model.max_elem() + 1;
            for k in 0..nz.max(1) {
                for j in 0..ny {
                    for i in 0..nx {
                        if areas {
                            let c = vec![idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0)];
                            self.add_element(eid, c, true);
                            eid += 1;
                            ne += 1;
                            continue;
                        }
                        let c = [
                            idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                            idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
                        ];
                        if kind == Kind::Solid187 {
                            for t in [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]] {
                                let mut tn = vec![c[t[0]], c[t[1]], c[t[2]], c[t[3]]];
                                let p: Vec<[f64; 3]> = tn.iter().map(|&n| self.model.node(n).unwrap()).collect();
                                let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                                let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
                                let cc = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
                                let vol = a[0] * (b[1] * cc[2] - b[2] * cc[1]) - a[1] * (b[0] * cc[2] - b[2] * cc[0]) + a[2] * (b[0] * cc[1] - b[1] * cc[0]);
                                if vol < 0. {
                                    tn.swap(1, 2);
                                }
                                self.add_element(eid, tn, true);
                                eid += 1;
                                ne += 1;
                            }
                        } else {
                            self.add_element(eid, c.to_vec(), true);
                            eid += 1;
                            ne += 1;
                        }
                    }
                }
            }
            self.model.volumes[vi].meshed = true;
        }
        let nn = self.model.node_count() - nn0;
        nn0 = nn;
        let _ = nn0;
        self.echo(format!(" ** Meshing done: {ne} elements, {nn} nodes generated **"));
    }

    /// quadratic elements given with corner nodes only (E without EMORE) get mid-side nodes now
    fn complete_quadratic(&mut self) {
        let todo: Vec<(usize, Option<Kind>, Vec<usize>)> = self
            .model
            .elem_ids()
            .filter_map(|i| {
                let e = self.model.elems[i].as_ref().unwrap();
                let k = self.model.kind_of(e);
                let short = match k {
                    Some(Kind::Solid186) => e.nodes.len() == 8,
                    Some(Kind::Solid187) | Some(Kind::Plane183) => e.nodes.len() == 4,
                    _ => false,
                };
                if short {
                    Some((i, k, e.nodes.clone()))
                } else {
                    None
                }
            })
            .collect();
        for (i, k, n) in todo {
            let full = self.complete_midsides(k, n);
            if let Some(Some(e)) = self.model.elems.get_mut(i) {
                e.nodes = full;
            }
        }
    }

    fn solve(&mut self) {
        self.complete_quadratic();
        if !self.errors.is_empty() {
            let n = self.errors.len();
            self.error(format!("SOLVE skipped: {n} input error(s) found before SOLVE. Fix them and run again."));
            return;
        }
        let m = &self.model;
        let mut dofs = 0u8;
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        let mut ignored: BTreeMap<String, usize> = BTreeMap::new();
        for e in m.sel_elem_ids() {
            let el = m.elems[e].as_ref().unwrap();
            let nm = m.etypes.get(&el.etype).map(|t| t.name.clone()).unwrap_or_else(|| format!("TYPE {}", el.etype));
            if m.kind_of(el).is_some() {
                dofs |= crate::elements::elem_dof_mask(m, el);
                *kinds.entry(nm).or_insert(0) += 1;
            } else {
                *ignored.entry(nm).or_insert(0) += 1;
            }
        }
        let dims = if dofs & 0b100 != 0 { "3-D" } else { "2-D" };
        let dofl: Vec<&str> = (0..6).filter(|d| dofs & (1 << d) != 0).map(|d| DOF_NAMES[d]).collect();
        let solver_name = if self.settings.solver == SolverKind::Pcg { "PCG" } else { "SPARSE" };
        let mut s = String::new();
        let _ = writeln!(s, "\n *****  SOLVE    COMMAND  *****\n");
        let _ = writeln!(s, "                       S O L U T I O N   O P T I O N S\n");
        let _ = writeln!(s, "   PROBLEM DIMENSIONALITY. . . . . . . . . . . . .{dims}");
        let _ = writeln!(s, "   DEGREES OF FREEDOM. . . . . . {}", dofl.join("   "));
        let _ = writeln!(s, "   ANALYSIS TYPE . . . . . . . . . . . . . . . . .STATIC (STEADY-STATE)");
        let _ = writeln!(s, "   EQUATION SOLVER OPTION. . . . . . . . . . . . .{solver_name}");
        if self.settings.solver == SolverKind::Pcg {
            let _ = writeln!(s, "      TOLERANCE. . . . . . . . . . . . . . . . . .{:e}", self.settings.pcg_tol);
        }
        let _ = writeln!(s, "   GLOBALLY ASSEMBLED MATRIX . . . . . . . . . . .SYMMETRIC\n");
        for (k, v) in &kinds {
            let _ = writeln!(s, "   {v:>10} elements of type {k}");
        }
        for (k, v) in &ignored {
            let _ = writeln!(s, "   {v:>10} elements of type {k}  (not supported - IGNORED)");
        }
        self.print(s);
        if !ignored.is_empty() {
            let msg = format!(
                "{} selected element(s) of unsupported types were ignored in the solution: {}",
                ignored.values().sum::<usize>(),
                ignored.keys().cloned().collect::<Vec<_>>().join(", ")
            );
            self.warn(msg);
        }
        self.check_ces();
        if self.no_solve {
            self.print(" SOLVE recorded for export to the selected solver (not solved here).");
            self.solve_models.push(self.model.clone());
            return;
        }
        let mut logs = Vec::new();
        // constraint equations: enforced by stiff penalty couplings between slave and master
        // (energy kp/2 (u_s - u_m - c)^2, i.e. a coupling stiffness plus the force pair +/- kp c)
        let (res, ce_ids) = if self.model.ces.is_empty() {
            (solver::solve(&self.model, &self.settings, &mut |l| logs.push(l)), Vec::new())
        } else {
            let mut m2 = self.model.clone();
            let emax = m2.mats.values().filter_map(|p| p.get("EX").copied()).fold(0f64, f64::max).max(1.0);
            let kp = CE_PENALTY * emax;
            let (t_ux, t_uy, t_uz, rset) = (CE_ETYPE, CE_ETYPE + 1, CE_ETYPE + 2, CE_ETYPE);
            for (k, t) in [t_ux, t_uy, t_uz].iter().enumerate() {
                let mut ko = [0i32; 19];
                ko[2] = k as i32 + 1;
                m2.etypes.insert(*t, EType { name: "CE".into(), kind: Some(Kind::Spring), keyopt: ko });
            }
            m2.reals.insert(rset, vec![kp]);
            let mut ids = Vec::new();
            let mut eid = m2.max_elem() + 1;
            for ce in self.model.ces.clone() {
                if ce.dof > 2 {
                    continue;
                }
                if ce.master == 0 {
                    // single-term equation: the dependent DOF is prescribed
                    m2.disp.insert((ce.slave, ce.dof), ce.c);
                    continue;
                }
                m2.set_elem(eid, Element { etype: CE_ETYPE + ce.dof as usize, mat: 1, real: rset, sec: 1, nodes: vec![ce.master, ce.slave] });
                *m2.force.entry((ce.slave, ce.dof)).or_insert(0.) += kp * ce.c;
                *m2.force.entry((ce.master, ce.dof)).or_insert(0.) -= kp * ce.c;
                ids.push((eid, ce, kp));
                eid += 1;
            }
            self.print(format!("   {} constraint equation(s) enforced (coupling stiffness {:.3e})", ids.len(), kp));
            (solver::solve(&m2, &self.settings, &mut |l| logs.push(l)), ids)
        };
        for l in logs {
            self.print(l);
        }
        match res {
            Ok(r) => {
                // constraint force on the slave node (external force the constraint applies): -kp (u_s - u_m - c)
                let forces: Vec<(Ce, f64)> = ce_ids
                    .iter()
                    .map(|(eid, ce, kp)| {
                        let f = r.elem_forces.get(eid).and_then(|v| v.first().copied()).unwrap_or(0.);
                        (*ce, -(f - kp * ce.c))
                    })
                    .collect();
                self.ce_forces.push(forces);
                let st = &r.stats;
                let mut s = String::new();
                if st.solver == "SPARSE" {
                    let _ = writeln!(s, "\n SPARSE MATRIX DIRECT SOLVER.");
                    let _ = writeln!(s, "  Number of equations = {:>12},    Maximum wavefront = n/a", st.equations);
                } else {
                    let _ = writeln!(s, "\n PCG SOLVER (nodal block-Jacobi preconditioner)");
                    let _ = writeln!(s, "  Number of equations = {:>12},    Iterations = {}", st.equations, st.pcg_iterations);
                }
                if st.threads > 1 {
                    let _ = writeln!(s, "\tNumber of cores used (Shared Memory Parallel): {}", st.threads);
                } else {
                    let _ = writeln!(s, "\tNumber of cores used: 1");
                }
                let _ = writeln!(s, "\n   Time (seconds, wall clock)");
                let _ = writeln!(s, "     Setup (DOF numbering, sparsity) . . {:>10.3}", st.t_setup);
                let _ = writeln!(s, "     Element matrices + assembly . . . . {:>10.3}", st.t_assembly);
                let _ = writeln!(s, "     Equation solution . . . . . . . . . {:>10.3}", st.t_solve);
                let _ = writeln!(s, "     Reactions + nodal stresses. . . . . {:>10.3}", st.t_post);
                let _ = writeln!(s, "     TOTAL SOLVE . . . . . . . . . . . . {:>10.3}", st.t_total);
                let _ = writeln!(s, "\n   *** LOAD STEP     {}   SUBSTEP     1  COMPLETED.    CUM ITER =      1", self.results.len() + 1);
                let _ = writeln!(s, "   *** TIME =   1.00000         TIME INC =   1.00000      NEW TRIANG MATRIX");
                self.print(s);
                for w in r.warnings.clone() {
                    self.warn(w);
                }
                self.results.push(r);
                self.cur = Some(self.results.len() - 1);
            }
            Err(e) => self.error(format!("SOLUTION FAILED: {e}")),
        }
    }

    /// SOLVE-time checks of the constraint equations, as in pr7vbc (0x145c2b260)
    fn check_ces(&mut self) {
        let defined = |m: &Model, n: usize| n > 0 && m.node(n).is_some();
        let mut deleted_all_d = 0;
        let ces = std::mem::take(&mut self.model.ces);
        let mut keep = Vec::with_capacity(ces.len());
        for ce in ces {
            let nodes: Vec<usize> = if ce.master == 0 { vec![ce.slave] } else { vec![ce.slave, ce.master] };
            let undefined = nodes.iter().filter(|&&n| !defined(&self.model, n)).count();
            if undefined == nodes.len() {
                self.warn(format!("Constraint equation {} uses only undefined nodes, it has been deleted.", ce.n));
                continue;
            }
            for &n in &nodes {
                if !defined(&self.model, n) {
                    self.warn(format!("Constraint equation {} has unused node {}.", ce.n, n));
                }
            }
            let constrained = nodes.iter().filter(|&&n| self.model.disp.contains_key(&(n, ce.dof))).count();
            if constrained == nodes.len() {
                // per-equation wording seen in the original run; the count form (0x155ce2500) is used for many
                self.warn(format!(
                    "Constraint equation {} has specified degree of freedom (dof) constraint at all nodes/dof.  Constraint equation deleted.",
                    ce.n
                ));
                deleted_all_d += 1;
                continue;
            }
            if self.model.disp.contains_key(&(ce.slave, ce.dof)) {
                self.error(format!(
                    "Constraint equation {} has indepedent degree of freedom as the first degree of freedom.  This is not allowed.",
                    ce.n
                ));
            }
            keep.push(ce);
        }
        if deleted_all_d > 1 {
            self.warn(format!(
                "{deleted_all_d} constraint equations have specified degree of freedom (dof) constraint at all nodes/dof. These equations are deleted."
            ));
        }
        self.model.ces = keep;
    }

    pub fn sel_summary(&self) -> (usize, usize) {
        self.sel_count()
    }
}

/// NSEL/ESEL tolerance factors of the original (probe run tests/seltol/probe_orig.out)
const SEL_TOL_RANGE: f64 = 1.0e-8;
const SEL_TOL_VALUE: f64 = 0.005;
const SEL_TOL_ZERO: f64 = 1.0e-6;

/// coupling stiffness of a constraint equation = CE_PENALTY x largest EX (relative error ~ 1/CE_PENALTY)
const CE_PENALTY: f64 = 1.0e8;
/// internal element-type numbers used for the CE couplings (UX, UY, UZ)
const CE_ETYPE: usize = 900001;

fn fmt_num(v: f64) -> String {
    if v.fract() == 0. && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}
