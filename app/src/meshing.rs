//! Meshing tab helpers: small APDL snippets for geometry, mesh, materials, conditions and loads,
//! inserted into the right section of the code window (so hand edits stay possible).

#[derive(Clone, Copy, PartialEq)]
pub enum Section {
    /// /PREP7 ... FINISH  (element types, materials, geometry, mesh)
    Prep,
    /// /SOLU ... SOLVE    (constraints, loads, solver options)
    Solu,
    /// after the solution (/POST1 listings)
    Post,
}

fn head(l: &str) -> String {
    l.trim().split(',').next().unwrap_or("").trim().to_uppercase()
}

pub const SKELETON: &str = "/TITLE, New model\n/PREP7\nFINISH\n/SOLU\nANTYPE,STATIC\nSOLVE\nFINISH\n/POST1\nPRNSOL,U\nPRRSOL\n";

/// insert `snippet` (one or more lines) into `code` in the given section; returns the line where it went
pub fn insert(code: &mut String, sec: Section, snippet: &str) -> usize {
    if code.trim().is_empty() {
        *code = SKELETON.to_string();
    }
    let mut lines: Vec<String> = code.lines().map(|s| s.to_string()).collect();
    let snip: Vec<String> = snippet.trim_end_matches('\n').lines().map(|s| s.to_string()).collect();
    let find = |lines: &[String], from: usize, pred: &dyn Fn(&str) -> bool| (from..lines.len()).find(|&i| pred(&head(&lines[i])));
    let at = match sec {
        Section::Prep => {
            let p = match find(&lines, 0, &|h| h.starts_with("/PREP")) {
                Some(p) => p,
                None => {
                    lines.insert(0, "/PREP7".into());
                    lines.insert(1, "FINISH".into());
                    0
                }
            };
            find(&lines, p + 1, &|h| h.starts_with("FINI") || h.starts_with("/SOLU") || h.starts_with("/POST")).unwrap_or(lines.len())
        }
        Section::Solu => match find(&lines, 0, &|h| h.starts_with("SOLV")) {
            Some(s) => s,
            None => match find(&lines, 0, &|h| h.starts_with("/SOLU")) {
                Some(p) => {
                    let e = find(&lines, p + 1, &|h| h.starts_with("FINI") || h.starts_with("/POST")).unwrap_or(lines.len());
                    lines.insert(e, "SOLVE".into());
                    e
                }
                None => {
                    let e = find(&lines, 0, &|h| h.starts_with("/POST")).unwrap_or(lines.len());
                    lines.insert(e, "/SOLU".into());
                    lines.insert(e + 1, "SOLVE".into());
                    lines.insert(e + 2, "FINISH".into());
                    e + 1
                }
            },
        },
        Section::Post => {
            if find(&lines, 0, &|h| h.starts_with("/POST")).is_none() {
                lines.push("/POST1".into());
            }
            lines.len()
        }
    };
    for (k, l) in snip.into_iter().enumerate() {
        lines.insert(at + k, l);
    }
    *code = lines.join("\n") + "\n";
    at
}

/// replace (or add) the /TITLE line
pub fn set_title(code: &mut String, title: &str) {
    let mut lines: Vec<String> = code.lines().map(|s| s.to_string()).collect();
    match lines.iter().position(|l| head(l).starts_with("/TITL")) {
        Some(i) => lines[i] = format!("/TITLE, {title}"),
        None => lines.insert(0, format!("/TITLE, {title}")),
    }
    *code = lines.join("\n") + "\n";
}

pub fn axis_name(a: usize) -> &'static str {
    ["X", "Y", "Z"][a]
}

/// select the nodes on a plane, apply, restore the selection
pub fn on_plane(axis: usize, at: f64, body: &str) -> String {
    format!("NSEL,S,LOC,{},{}\n{}\nALLSEL", axis_name(axis), fmt(at), body.trim_end())
}

pub fn fmt(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e12 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inserts_in_sections() {
        let mut c = SKELETON.to_string();
        insert(&mut c, Section::Prep, "ET,1,SOLID185");
        insert(&mut c, Section::Solu, "D,1,ALL,0");
        insert(&mut c, Section::Post, "PRNSOL,S");
        let l: Vec<&str> = c.lines().collect();
        let p = l.iter().position(|x| *x == "ET,1,SOLID185").unwrap();
        assert_eq!(l[p - 1], "/PREP7");
        let d = l.iter().position(|x| *x == "D,1,ALL,0").unwrap();
        assert_eq!(l[d + 1], "SOLVE");
        assert_eq!(*l.last().unwrap(), "PRNSOL,S");
    }
}
