//! Built-in example inputs (plain MAPDL).

pub const EXAMPLES: [(&str, &str); 4] = [
    ("Solid cantilever (SOLID185, meshed block)", SOLID_BLOCK),
    ("Cantilever beam (BEAM188)", BEAM),
    ("Plane truss (LINK180)", TRUSS),
    ("Plate in tension (PLANE182, pressure)", PLATE),
];

pub const SOLID_BLOCK: &str = r#"/TITLE, Solid cantilever 200 x 20 x 20 mm, end load
! Hand check: tip deflection PL^3/(3EI) + shear = about 0.96 mm (UY)
/PREP7
ET,1,SOLID185
KEYOPT,1,2,3            ! simplified enhanced strain
MP,EX,1,210000          ! N/mm2
MP,PRXY,1,0.3
MP,DENS,1,7.85e-9
BLOCK,0,200,0,20,0,20
ESIZE,4
VMESH,ALL
FINISH
/SOLU
ANTYPE,STATIC
NSEL,S,LOC,X,0
D,ALL,ALL,0
NSEL,S,LOC,X,200
*GET,nload,NODE,0,COUNT
F,ALL,FY,-1000/nload    ! 1 kN total
ALLSEL
SOLVE
FINISH
/POST1
*GET,ntip,NODE,0,NUM,MAX
PLNSOL,U,SUM
*GET,umax,PLNSOL,0,MAX
PLNSOL,S,EQV
*GET,smax,PLNSOL,0,MAX
NSEL,S,LOC,X,0
PRRSOL
ALLSEL
/COM, Maximum displacement and von Mises stress:
*VWRITE,umax,smax
(2E14.5)
"#;

pub const BEAM: &str = r#"/TITLE, Cantilever beam BEAM188 - tip load
! Hand check: UZ tip = P L^3 / (3 E Iy) = -1.488 mm (bending), shear adds ~0.3 %
/PREP7
ET,1,BEAM188
MP,EX,1,210000
MP,PRXY,1,0.3
SECTYPE,1,BEAM,RECT
SECDATA,20,40           ! B (local y) = 20, H (local z) = 40
L = 1000
NE = 40
*DO,i,1,NE+1
  N,i,(i-1)*L/NE
*ENDDO
*DO,i,1,NE
  E,i,i+1
*ENDDO
FINISH
/SOLU
D,1,ALL,0
F,NE+1,FZ,-100
SOLVE
FINISH
/POST1
PRNSOL,U
PRRSOL
*GET,tip,NODE,NE+1,U,Z
/COM, Tip deflection UZ:
*VWRITE,tip
(E14.5)
"#;

pub const TRUSS: &str = r#"/TITLE, Plane truss LINK180
/PREP7
ET,1,LINK180
SECTYPE,1,LINK
SECDATA,100             ! area mm2
MP,EX,1,200000
N,1,0,0
N,2,1000,0
N,3,2000,0
N,4,3000,0
N,5,500,800
N,6,1500,800
N,7,2500,800
E,1,2
E,2,3
E,3,4
E,5,6
E,6,7
E,1,5
E,5,2
E,2,6
E,6,3
E,3,7
E,7,4
FINISH
/SOLU
D,ALL,UZ,0
D,1,UX,0,,,,UY
D,4,UY,0
F,2,FY,-10000
F,3,FY,-10000
SOLVE
FINISH
/POST1
PRNSOL,U
PRRSOL
PRESOL
"#;

pub const PLATE: &str = r#"/TITLE, Plate in tension PLANE182 (plane stress, thickness 5)
! Hand check: SX = 10 MPa everywhere, UX(end) = 10*400/70000 = 0.0571 mm
/PREP7
ET,1,PLANE182,,,3       ! KEYOPT(3)=3: plane stress with thickness
R,1,5
MP,EX,1,70000
MP,PRXY,1,0.33
RECTNG,0,400,0,100
ESIZE,10
AMESH,ALL
FINISH
/SOLU
NSEL,S,LOC,X,0
D,ALL,UX,0
NSEL,R,LOC,Y,0
D,ALL,UY,0
NSEL,S,LOC,X,400
SF,ALL,PRES,-10         ! negative pressure = tension
ALLSEL
SOLVE
FINISH
/POST1
PRNSOL,S,PRIN
NSEL,S,LOC,X,0
PRRSOL
ALLSEL
"#;

/// A scalable test model for speed comparisons: a 20-node brick block.
pub fn benchmark(div: usize, quadratic: bool) -> String {
    let et = if quadratic { "SOLID186" } else { "SOLID185" };
    format!(
        r#"/TITLE, Speed test - {et} block {div}x{d2}x{d2}
/PREP7
ET,1,{et}
MP,EX,1,210000
MP,PRXY,1,0.3
BLOCK,0,{len},0,100,0,100
ESIZE,{sz}
VMESH,ALL
FINISH
/SOLU
NSEL,S,LOC,X,0
D,ALL,ALL,0
NSEL,S,LOC,X,{len}
SF,ALL,PRES,10
ALLSEL
SOLVE
FINISH
/POST1
PLNSOL,U,SUM
*GET,umax,PLNSOL,0,MAX
*VWRITE,umax
(E14.5)
"#,
        d2 = (div / 2).max(1),
        len = 200,
        sz = 200.0 / div as f64,
    )
}
