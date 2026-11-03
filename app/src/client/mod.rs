//! fem-client: prepares, runs and compares finite-element jobs on several solver engines
//! (Code_Aster, FEniCSx, Ansys MAPDL) from one neutral model. It drives the engines through their
//! own interfaces and reimplements no solver.

pub mod adapters;
pub mod cli;
pub mod compare;
pub mod export;
pub mod jobs;
pub mod local;
pub mod model;
pub mod ui;

use sha2::{Digest, Sha256};

pub fn sha256_hex(b: &[u8]) -> String {
    let d = Sha256::digest(b);
    d.iter().map(|x| format!("{x:02x}")).collect()
}

/// the defaults the window and the command line start from
pub fn default_job() -> model::Job {
    model::Job {
        cell: model::CellParams { kind: model::CellKind::RotatingSquares, a: 10.0, theta_deg: 15.0, t: 1.0 },
        material: model::Material { e: 2500.0, nu: 0.35, thickness: 1.0 },
        mesh: model::MeshSettings { size: 0.4, order: 2, seed: 1 },
        load: model::LoadCase::Homogenize { strain: 1e-3 },
    }
}
