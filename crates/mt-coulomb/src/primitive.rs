//! SPEX `primitive` / `primitivef` radial primitives and the intra-sphere Poisson kernel.

use crate::CoulombError;
use libmuffintin_core::ExponentialMesh;
use std::f64::consts::PI;

const LAGRANGE: [[f64; 7]; 6] = [
    [
        19_087.0, 65_112.0, -46_461.0, 37_504.0, -20_211.0, 6_312.0, -863.0,
    ],
    [
        -863.0, 25_128.0, 46_989.0, -16_256.0, 7_299.0, -2_088.0, 271.0,
    ],
    [
        271.0, -2_760.0, 30_819.0, 37_504.0, -6_771.0, 1_608.0, -191.0,
    ],
    [
        -191.0, 1_608.0, -6_771.0, 37_504.0, 30_819.0, -2_760.0, 271.0,
    ],
    [
        271.0, -2_088.0, 7_299.0, -16_256.0, 46_989.0, 25_128.0, -863.0,
    ],
    [
        -863.0, 6_312.0, -20_211.0, 37_504.0, -46_461.0, 65_112.0, 19_087.0,
    ],
];

/// SPEX `primitive` (`numerics.f:267-345`).
///
/// Outward: $\int_0^r f(r')\,\mathrm{d}r'$. Inward: $\int_r^R f(r')\,\mathrm{d}r'$.
pub fn radial_primitive(
    mesh: &ExponentialMesh,
    values: &[f64],
    inward: bool,
) -> Result<Vec<f64>, CoulombError> {
    if values.len() != mesh.len() {
        return Err(CoulombError::from(
            libmuffintin_core::MeshError::LengthMismatch {
                mesh: mesh.len(),
                values: values.len(),
            },
        ));
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CoulombError::NonFiniteRadial { index });
    }
    let n = mesh.len();
    let mut primf = vec![0.0; n];
    let mut h = mesh.increment();
    let mut f = values.to_vec();
    let r1 = if inward {
        h = -h;
        f.reverse();
        mesh.last().get()
    } else {
        mesh.first().get()
    };

    let mut intgr = 0.0;
    if h > 0.0 && f[0] * f[1] > 1.0e-20 {
        let exponent = (f[1] / f[0]).ln() / h;
        intgr = if exponent <= -1.0 {
            r1 * f[0] / 2.0
        } else {
            r1 * f[0] / (exponent + 1.0)
        };
    }
    primf[0] = intgr;
    let dr = h.exp();
    let h1 = h / 60_480.0;
    let mut n0 = 0;
    loop {
        if n0 + 6 >= n {
            break;
        }
        let mut radius = [0.0; 7];
        radius[0] = if inward {
            mesh.radii()[n - 1 - n0].get()
        } else {
            mesh.radii()[n0].get()
        };
        for i in 1..7 {
            radius[i] = radius[i - 1] * dr;
        }
        let mut fr = [0.0; 7];
        for i in 0..7 {
            fr[i] = f[n0 + i] * radius[i];
        }
        for i in 0..6 {
            let mut acc = 0.0;
            for k in 0..7 {
                acc += LAGRANGE[i][k] * fr[k];
            }
            intgr += h1 * acc;
            if primf[n0 + 1 + i] == 0.0 {
                primf[n0 + 1 + i] = intgr;
            }
        }
        if n0 + 12 < n {
            n0 += 6;
        } else if n0 + 6 < n - 1 {
            intgr = primf[n - 7];
            n0 = n - 7;
        } else {
            break;
        }
    }
    if inward {
        primf.reverse();
        for value in &mut primf {
            *value = -*value;
        }
    }
    Ok(primf)
}

/// Intra-sphere radial Poisson kernel (`coulombmatrix.f:339-346`).
///
/// `b` is the SPEX mixed-basis radial (`basm`, already including one factor of $r$).
pub fn intra_sphere_poisson(
    l: u32,
    mesh: &ExponentialMesh,
    left: &[f64],
    right: &[f64],
) -> Result<f64, CoulombError> {
    let n = mesh.len();
    if left.len() != n || right.len() != n {
        return Err(CoulombError::from(
            libmuffintin_core::MeshError::LengthMismatch {
                mesh: n,
                values: left.len().min(right.len()),
            },
        ));
    }
    let radii = mesh.radii();
    let mut f_out = vec![0.0; n];
    let mut f_in = vec![0.0; n];
    for i in 0..n {
        let r = radii[i].get();
        let power = r.powi(l as i32);
        f_out[i] = right[i] * power * r;
        f_in[i] = if l == 0 { right[i] } else { right[i] / power };
    }
    let prim_out = radial_primitive(mesh, &f_out, false)?;
    let prim_in = radial_primitive(mesh, &f_in, true)?;
    let mut integrand = vec![0.0; n];
    for i in 0..n {
        let r = radii[i].get();
        let power = r.powi(l as i32);
        let first = if l == 0 {
            prim_out[i]
        } else {
            prim_out[i] / power
        };
        let second = prim_in[i] * power * r;
        integrand[i] = left[i] * (first + second);
    }
    Ok(4.0 * PI / (2.0 * f64::from(l) + 1.0) * mesh.integrate(&integrand)?)
}
