//! MAPDL-style arithmetic expressions: numbers, parameters, arrays,
//! + - * / ** ^, parentheses and functions.

pub trait EvalCtx {
    /// Scalar parameter lookup (name is upper case).
    fn var(&self, name: &str) -> Option<f64>;
    /// Array element or model function, e.g. ARR(2), UX(10), NODE(0,1,0).
    fn call(&self, name: &str, args: &[f64]) -> Option<f64>;
    /// true when *AFUN,DEG is active.
    fn degrees(&self) -> bool;
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Id(String),
    Op(char),
    Pow,
    LP,
    RP,
    Comma,
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let st = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            if i < b.len() && matches!(b[i], 'e' | 'E' | 'd' | 'D') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == '+' || b[i] == '-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let txt: String = b[st..i].iter().collect::<String>().replace(['d', 'D'], "e");
            let v: f64 = txt.parse().map_err(|_| format!("bad number '{txt}'"))?;
            out.push(Tok::Num(v));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(Tok::Id(b[st..i].iter().collect::<String>().to_uppercase()));
            continue;
        }
        match c {
            '*' if i + 1 < b.len() && b[i + 1] == '*' => {
                out.push(Tok::Pow);
                i += 2;
                continue;
            }
            '^' => out.push(Tok::Pow),
            '+' | '-' | '*' | '/' => out.push(Tok::Op(c)),
            '(' => out.push(Tok::LP),
            ')' => out.push(Tok::RP),
            ',' => out.push(Tok::Comma),
            _ => return Err(format!("unexpected '{c}'")),
        }
        i += 1;
    }
    Ok(out)
}

struct P<'a> {
    t: Vec<Tok>,
    i: usize,
    ctx: &'a dyn EvalCtx,
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        while let Some(Tok::Op(c)) = self.peek().cloned() {
            if c == '+' || c == '-' {
                self.i += 1;
                let r = self.term()?;
                v = if c == '+' { v + r } else { v - r };
            } else {
                break;
            }
        }
        Ok(v)
    }
    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        while let Some(Tok::Op(c)) = self.peek().cloned() {
            if c == '*' || c == '/' {
                self.i += 1;
                let r = self.unary()?;
                v = if c == '*' { v * r } else { v / r };
            } else {
                break;
            }
        }
        Ok(v)
    }
    fn unary(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(Tok::Op('-')) => {
                self.i += 1;
                Ok(-self.unary()?)
            }
            Some(Tok::Op('+')) => {
                self.i += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }
    fn power(&mut self) -> Result<f64, String> {
        let b = self.atom()?;
        if let Some(Tok::Pow) = self.peek() {
            self.i += 1;
            let e = self.unary()?; // right associative
            return Ok(b.powf(e));
        }
        Ok(b)
    }
    fn atom(&mut self) -> Result<f64, String> {
        match self.t.get(self.i).cloned() {
            Some(Tok::Num(v)) => {
                self.i += 1;
                Ok(v)
            }
            Some(Tok::LP) => {
                self.i += 1;
                let v = self.expr()?;
                if self.peek() != Some(&Tok::RP) {
                    return Err("missing ')'".into());
                }
                self.i += 1;
                Ok(v)
            }
            Some(Tok::Id(name)) => {
                self.i += 1;
                if self.peek() == Some(&Tok::LP) {
                    self.i += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RP) {
                        loop {
                            args.push(self.expr()?);
                            match self.peek() {
                                Some(Tok::Comma) => self.i += 1,
                                Some(Tok::RP) => break,
                                _ => return Err("expected ',' or ')'".into()),
                            }
                        }
                    }
                    self.i += 1;
                    if let Some(v) = builtin(&name, &args, self.ctx.degrees()) {
                        return Ok(v);
                    }
                    return self
                        .ctx
                        .call(&name, &args)
                        .ok_or_else(|| format!("unknown function or array '{name}'"));
                }
                match name.as_str() {
                    "PI" => return Ok(std::f64::consts::PI),
                    _ => {}
                }
                self.ctx
                    .var(&name)
                    .ok_or_else(|| format!("undefined parameter '{name}'"))
            }
            other => Err(format!("unexpected {:?}", other)),
        }
    }
}

fn builtin(name: &str, a: &[f64], deg: bool) -> Option<f64> {
    let to_rad = |x: f64| if deg { x.to_radians() } else { x };
    let from_rad = |x: f64| if deg { x.to_degrees() } else { x };
    let x = a.first().copied().unwrap_or(0.0);
    Some(match (name, a.len()) {
        ("ABS", 1) => x.abs(),
        ("SIGN", 2) => x.abs() * if a[1] < 0.0 { -1.0 } else { 1.0 },
        ("CXABS", 2) => (a[0] * a[0] + a[1] * a[1]).sqrt(),
        ("EXP", 1) => x.exp(),
        ("LOG", 1) => x.ln(),
        ("LOG10", 1) => x.log10(),
        ("SQRT", 1) => x.sqrt(),
        ("NINT", 1) => x.round(),
        ("INT", 1) => x.trunc(),
        ("MOD", 2) => x % a[1],
        ("SIN", 1) => to_rad(x).sin(),
        ("COS", 1) => to_rad(x).cos(),
        ("TAN", 1) => to_rad(x).tan(),
        ("SINH", 1) => x.sinh(),
        ("COSH", 1) => x.cosh(),
        ("TANH", 1) => x.tanh(),
        ("ASIN", 1) => from_rad(x.asin()),
        ("ACOS", 1) => from_rad(x.acos()),
        ("ATAN", 1) => from_rad(x.atan()),
        ("ATAN2", 2) => from_rad(x.atan2(a[1])),
        ("MIN", n) if n >= 1 => a.iter().cloned().fold(f64::INFINITY, f64::min),
        ("MAX", n) if n >= 1 => a.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        _ => return None,
    })
}

pub fn eval(s: &str, ctx: &dyn EvalCtx) -> Result<f64, String> {
    let t = lex(s)?;
    if t.is_empty() {
        return Err("empty".into());
    }
    let mut p = P { t, i: 0, ctx };
    let v = p.expr()?;
    if p.i != p.t.len() {
        return Err(format!("unexpected text in '{s}'"));
    }
    Ok(v)
}
