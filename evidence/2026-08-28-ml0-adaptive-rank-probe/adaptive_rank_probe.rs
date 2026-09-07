//! M-L0 adaptive-weight rank gate.
//!
//! Controlled A/B of public `muffintin_thc` weighted QRCP on the Sm 3x3x3
//! NUF1 v2 adaptive dump. True adaptive quadrature weights versus uniform
//! weights, identical deterministic candidate subsets at M=512 and M=2048.
//!
//! This binary tests adaptive weighted selection only. It does not claim
//! all-q Weinert, canonical-q / Umklapp validation, or M-L completion.
//! Pair columns use semantic [`PairColumnLayout::encode`], not the
//! experiment's packed 12x12 target x occupied flattening.

use anyhow::{Context, Result, bail};
use muffintin_thc::{
    BlochOrbitals, KMesh, PairBlock, PairColumnLayout, UmklappGauge, evaluate_pair_block,
    pivots_from_pair_blocks, validate_quadrature_weights,
};
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAGIC: i32 = 0x4e55_4631; // NUF1
const VERSION: i32 = 2;
const THRESHOLDS: [f64; 4] = [1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3];
const M_VALUES: [usize; 2] = [512, 2048];
const BASELINE_M512_UNIFORM_RANK_3E3: usize = 113;

struct AdaptiveDump {
    nk: usize,
    nspin: usize,
    bando: usize,
    nband: usize,
    npts: usize,
    nrad: usize,
    nang: usize,
    ninter: usize,
    vol: f64,
    lat: [f64; 9],
    kpt: Vec<[f64; 3]>,
    occupations: Vec<f64>,
    points: Vec<[f64; 3]>,
    quadrature: Vec<f64>,
    wavefunctions_large: Vec<Complex64>,
    wavefunctions_small: Vec<Complex64>,
}

fn read_exact<const N: usize>(file: &mut BufReader<File>) -> Result<[u8; N]> {
    let mut bytes = [0_u8; N];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_i32(file: &mut BufReader<File>) -> Result<i32> {
    Ok(i32::from_le_bytes(read_exact::<4>(file)?))
}

fn read_f64(file: &mut BufReader<File>) -> Result<f64> {
    Ok(f64::from_le_bytes(read_exact::<8>(file)?))
}

fn read_f64_vec(file: &mut BufReader<File>, count: usize) -> Result<Vec<f64>> {
    let mut bytes = vec![0_u8; 8 * count];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|part| f64::from_le_bytes(part.try_into().expect("8-byte chunk")))
        .collect())
}

fn read_adaptive(path: &Path) -> Result<AdaptiveDump> {
    let mut file =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    let header: Vec<i32> = (0..11)
        .map(|_| read_i32(&mut file))
        .collect::<Result<_>>()?;
    if header[0] != MAGIC || header[1] != VERSION || header[10] != 2 {
        bail!("unsupported adaptive dump header: {header:?}");
    }
    let nk = header[2] as usize;
    let nspin = header[3] as usize;
    let bando = header[4] as usize;
    let nband = header[5] as usize;
    let npts = header[6] as usize;
    let nrad = header[7] as usize;
    let nang = header[8] as usize;
    let ninter = header[9] as usize;
    if nk == 0 || nspin == 0 || bando == 0 || nband < bando || npts == 0 {
        bail!("inconsistent NUF1 dimensions: {header:?}");
    }
    if nrad.checked_mul(nang).context("nrad*nang overflow")? >= npts {
        bail!("nrad*nang={nrad}*{nang} does not leave interstitial samples");
    }
    let vol = read_f64(&mut file)?;
    let lat_raw = read_f64_vec(&mut file, 9)?;
    let mut lat = [0.0; 9];
    lat.copy_from_slice(&lat_raw);
    let _rlat = read_f64_vec(&mut file, 9)?;
    let kraw = read_f64_vec(&mut file, 3 * nk)?;
    let kpt = (0..nk)
        .map(|k| [kraw[3 * k], kraw[3 * k + 1], kraw[3 * k + 2]])
        .collect();
    let occupations = read_f64_vec(&mut file, nk * bando * nspin)?;
    let praw = read_f64_vec(&mut file, 3 * npts)?;
    let points = (0..npts)
        .map(|p| [praw[3 * p], praw[3 * p + 1], praw[3 * p + 2]])
        .collect();
    let quadrature = read_f64_vec(&mut file, npts)?;

    let mut wavefunctions_large = vec![Complex64::new(0.0, 0.0); nspin * nk * nband * npts];
    let mut wavefunctions_small = vec![Complex64::new(0.0, 0.0); nspin * nk * nband * npts];
    let mut bytes = vec![0_u8; 16 * nband * npts];
    for spin in 0..nspin {
        for k in 0..nk {
            for component in 0..2 {
                file.read_exact(&mut bytes)?;
                for point in 0..npts {
                    for band in 0..nband {
                        let src = 16 * (band + nband * point);
                        let re = f64::from_le_bytes(bytes[src..src + 8].try_into()?);
                        let im = f64::from_le_bytes(bytes[src + 8..src + 16].try_into()?);
                        // nband-strided wavefunction index; never use bando here.
                        let dst = point + npts * (band + nband * (k + nk * spin));
                        if component == 0 {
                            wavefunctions_large[dst] = Complex64::new(re, im);
                        } else {
                            wavefunctions_small[dst] = Complex64::new(re, im);
                        }
                    }
                }
            }
        }
    }
    Ok(AdaptiveDump {
        nk,
        nspin,
        bando,
        nband,
        npts,
        nrad,
        nang,
        ninter,
        vol,
        lat,
        kpt,
        occupations,
        points,
        quadrature,
        wavefunctions_large,
        wavefunctions_small,
    })
}

impl AdaptiveDump {
    fn n_mt(&self) -> usize {
        self.nrad * self.nang
    }

    fn orbital(
        store: &[Complex64],
        npts: usize,
        nband: usize,
        nk: usize,
        spin: usize,
        k: usize,
        band: usize,
        point: usize,
    ) -> Complex64 {
        store[point + npts * (band + nband * (k + nk * spin))]
    }

    fn occupation(&self, spin: usize, k: usize, band: usize) -> f64 {
        // bando-strided occupation index; never use nband here.
        self.occupations[k + self.nk * (band + self.bando * spin)]
    }
}

fn strided_indices(start: usize, count: usize, take: usize) -> Result<Vec<usize>> {
    if take == 0 || take > count {
        bail!("cannot take {take} strided samples from {count} points");
    }
    Ok((0..take).map(|i| start + i * count / take).collect())
}

/// Deterministic MT-then-interstitial subset. Identical for both weight arms.
fn candidate_subset(n_mt: usize, n_int: usize, m: usize) -> Result<Vec<usize>> {
    let npts = n_mt + n_int;
    if m == 0 || m > npts {
        bail!("M={m} is outside 1..={npts}");
    }
    let n_mt_c = ((m as u128 * n_mt as u128) / npts as u128) as usize;
    let n_mt_c = n_mt_c.clamp(1, n_mt.min(m.saturating_sub(1)));
    let n_int_c = m - n_mt_c;
    if n_int_c == 0 || n_int_c > n_int {
        bail!("interstitial take {n_int_c} is outside 1..={n_int}");
    }
    let mut idx = strided_indices(0, n_mt, n_mt_c)?;
    idx.extend(strided_indices(n_mt, n_int, n_int_c)?);
    debug_assert_eq!(idx.len(), m);
    Ok(idx)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let x = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let i = x.floor() as usize;
    let f = x - i as f64;
    if i + 1 < sorted.len() {
        sorted[i] * (1.0 - f) + sorted[i + 1] * f
    } else {
        sorted[i]
    }
}

struct WeightStats {
    min: f64,
    p10: f64,
    median: f64,
    p90: f64,
    max: f64,
    mean: f64,
}

fn weight_stats(values: &[f64]) -> WeightStats {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean = if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    WeightStats {
        min: sorted.first().copied().unwrap_or(f64::NAN),
        p10: percentile(&sorted, 0.10),
        median: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        max: sorted.last().copied().unwrap_or(f64::NAN),
        mean,
    }
}

fn subsampled_orbitals(
    dump: &AdaptiveDump,
    spin: usize,
    candidates: &[usize],
    small: bool,
) -> Result<BlochOrbitals> {
    let n_pts = candidates.len();
    let n_k = dump.nk;
    let n_orb = dump.nband;
    let mut values = vec![Complex64::new(0.0, 0.0); n_pts * n_k * n_orb];
    let store = if small {
        &dump.wavefunctions_small
    } else {
        &dump.wavefunctions_large
    };
    for (local, &point) in candidates.iter().enumerate() {
        for k in 0..n_k {
            for orb in 0..n_orb {
                values[(local * n_k + k) * n_orb + orb] = AdaptiveDump::orbital(
                    store, dump.npts, dump.nband, dump.nk, spin, k, orb, point,
                );
            }
        }
    }
    Ok(BlochOrbitals::new(n_pts, n_k, n_orb, values)?)
}

fn q0_ls_pair_block(
    dump: &AdaptiveDump,
    spin: usize,
    candidates: &[usize],
    points: &[[f64; 3]],
    mesh: &KMesh,
) -> Result<PairBlock> {
    let large = subsampled_orbitals(dump, spin, candidates, false)?;
    let small = subsampled_orbitals(dump, spin, candidates, true)?;
    let block_l = evaluate_pair_block(&large, points, mesh, 0, None, UmklappGauge::Canonical)?;
    let block_s = evaluate_pair_block(&small, points, mesh, 0, None, UmklappGauge::Canonical)?;
    if block_l.layout != block_s.layout || block_l.n_points != block_s.n_points {
        bail!("large/small pair blocks disagree for spin {spin}");
    }
    let values: Vec<Complex64> = block_l
        .values()
        .iter()
        .zip(block_s.values())
        .map(|(left, right)| left + right)
        .collect();
    Ok(PairBlock::new(0, block_l.n_points, block_l.layout, values)?)
}

fn rank_from_diag(diag: &[f64], thresh: f64) -> Result<usize> {
    let r0 = match diag.first() {
        Some(&value) if value.is_finite() && value > 0.0 => value,
        _ => bail!("degenerate QRCP leading residual"),
    };
    let kept = diag
        .iter()
        .take_while(|&&value| value >= thresh * r0)
        .count();
    if kept == 0 {
        bail!("threshold {thresh} kept no pivots");
    }
    Ok(kept)
}

fn peak_rss_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

struct ArmRow {
    m: usize,
    arm: &'static str,
    threshold: f64,
    rank: usize,
    n_keep: usize,
    n_candidates: usize,
    n_mt_cand: usize,
    n_int_cand: usize,
    n_mt_piv: usize,
    n_int_piv: usize,
    mt_frac_piv: f64,
    w_cand: WeightStats,
    w_piv: WeightStats,
    diag0: f64,
    saturated: bool,
    low_decile_frac: f64,
    wall_s: f64,
}

fn run_arm(
    dump: &AdaptiveDump,
    mesh: &KMesh,
    m: usize,
    candidates: &[usize],
    true_weights: &[f64],
    arm: &'static str,
    weights: &[f64],
    n_mt: usize,
) -> Result<Vec<ArmRow>> {
    validate_quadrature_weights(weights)?;
    let start = Instant::now();
    let points: Vec<[f64; 3]> = candidates.iter().map(|&p| dump.points[p]).collect();
    let mut blocks = Vec::with_capacity(dump.nspin);
    for spin in 0..dump.nspin {
        blocks.push(q0_ls_pair_block(dump, spin, candidates, &points, mesh)?);
    }
    let layout = blocks[0].layout;
    let expected = PairColumnLayout::new(dump.nk, dump.nband, None);
    if layout != expected {
        bail!("layout {layout:?} != semantic {expected:?}");
    }
    let encoded = layout.encode(1, 2, 3);
    let semantic = 1 * dump.nband * dump.nband + 2 * dump.nband + 3;
    if encoded != semantic {
        bail!("encode(1,2,3)={encoded} != {semantic}; 12x12 packing would not match");
    }
    let n_keep = m;
    let (pivots, diag) = pivots_from_pair_blocks(&blocks, weights, n_keep)?;
    drop(blocks);
    let wall_s = start.elapsed().as_secs_f64();
    let diag0 = diag[0];
    let n_mt_cand = candidates.iter().filter(|&&p| p < n_mt).count();
    let n_int_cand = candidates.len() - n_mt_cand;
    let w_cand = weight_stats(true_weights);
    let mut sorted_true = true_weights.to_vec();
    sorted_true.sort_by(|a, b| a.total_cmp(b));
    let decile = percentile(&sorted_true, 0.10);

    let mut rows = Vec::new();
    for &threshold in &THRESHOLDS {
        let rank = rank_from_diag(&diag, threshold)?.min(n_keep);
        let chosen = &pivots[..rank.min(pivots.len())];
        let n_mt_piv = chosen
            .iter()
            .filter(|&&local| candidates[local] < n_mt)
            .count();
        let n_int_piv = chosen.len() - n_mt_piv;
        let pivot_true: Vec<f64> = chosen.iter().map(|&local| true_weights[local]).collect();
        let w_piv = weight_stats(&pivot_true);
        let low_decile = if chosen.is_empty() {
            f64::NAN
        } else {
            chosen
                .iter()
                .filter(|&&local| true_weights[local] <= decile)
                .count() as f64
                / chosen.len() as f64
        };
        rows.push(ArmRow {
            m,
            arm,
            threshold,
            rank,
            n_keep,
            n_candidates: candidates.len(),
            n_mt_cand,
            n_int_cand,
            n_mt_piv,
            n_int_piv,
            mt_frac_piv: n_mt_piv as f64 / chosen.len().max(1) as f64,
            w_cand: WeightStats {
                min: w_cand.min,
                p10: w_cand.p10,
                median: w_cand.median,
                p90: w_cand.p90,
                max: w_cand.max,
                mean: w_cand.mean,
            },
            w_piv,
            diag0,
            saturated: rank >= n_keep,
            low_decile_frac: low_decile,
            wall_s,
        });
    }
    Ok(rows)
}

fn write_csv(path: &Path, rows: &[ArmRow]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "m,weight_arm,threshold,rank,n_keep,n_candidates,n_mt_cand,n_int_cand,\
n_mt_piv,n_int_piv,mt_frac_piv,w_cand_min,w_cand_p10,w_cand_median,w_cand_p90,\
w_cand_max,w_cand_mean,w_piv_min,w_piv_p10,w_piv_median,w_piv_p90,w_piv_max,\
w_piv_mean,diag0,saturated,low_decile_frac,wall_s"
    )?;
    for row in rows {
        writeln!(
            out,
            "{},{},{:.8e},{},{},{},{},{},{},{},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.8},{:.6}",
            row.m,
            row.arm,
            row.threshold,
            row.rank,
            row.n_keep,
            row.n_candidates,
            row.n_mt_cand,
            row.n_int_cand,
            row.n_mt_piv,
            row.n_int_piv,
            row.mt_frac_piv,
            row.w_cand.min,
            row.w_cand.p10,
            row.w_cand.median,
            row.w_cand.p90,
            row.w_cand.max,
            row.w_cand.mean,
            row.w_piv.min,
            row.w_piv.p10,
            row.w_piv.median,
            row.w_piv.p90,
            row.w_piv.max,
            row.w_piv.mean,
            row.diag0,
            u8::from(row.saturated),
            row.low_decile_frac,
            row.wall_s
        )?;
    }
    Ok(())
}

fn gate_verdict(rows: &[ArmRow]) -> (bool, Vec<String>) {
    let mut notes = Vec::new();
    let mut pass = true;
    for &m in &M_VALUES {
        let tight_true = rows
            .iter()
            .find(|r| r.m == m && r.arm == "true" && (r.threshold - 3.0e-3).abs() < 1.0e-15);
        let tight_uniform = rows
            .iter()
            .find(|r| r.m == m && r.arm == "uniform" && (r.threshold - 3.0e-3).abs() < 1.0e-15);
        let Some(true_row) = tight_true else {
            notes.push(format!("M={m}: missing true 3e-3 row"));
            pass = false;
            continue;
        };
        let Some(uni_row) = tight_uniform else {
            notes.push(format!("M={m}: missing uniform 3e-3 row"));
            pass = false;
            continue;
        };
        if true_row.saturated {
            notes.push(format!(
                "M={m}: true-weight rank {} saturates n_keep={}",
                true_row.rank, true_row.n_keep
            ));
            pass = false;
        }
        let ratio = true_row.rank as f64 / uni_row.rank.max(1) as f64;
        // Coarse A/B: not a physics tolerance. Factor-of-four is "sharp".
        if ratio < 0.25 || ratio > 4.0 {
            notes.push(format!(
                "M={m}: true rank {} sharply diverges from uniform {} (ratio {ratio:.3})",
                true_row.rank, uni_row.rank
            ));
            pass = false;
        }
        if true_row.low_decile_frac > 0.5 {
            notes.push(format!(
                "M={m}: {:.1}% of 3e-3 true-weight pivots sit in the lowest candidate-weight decile",
                100.0 * true_row.low_decile_frac
            ));
            pass = false;
        }
        if true_row.w_piv.mean <= true_row.w_cand.min * 1.01 {
            notes.push(format!(
                "M={m}: pivot mean weight {:.3e} collapsed onto candidate min {:.3e}",
                true_row.w_piv.mean, true_row.w_cand.min
            ));
            pass = false;
        }
        notes.push(format!(
            "M={m} 3e-3: true_rank={} uniform_rank={} ratio={ratio:.3} sat={} mt_frac={:.3} low_decile={:.3} w_piv_mean/w_cand_mean={:.3}",
            true_row.rank,
            uni_row.rank,
            true_row.saturated,
            true_row.mt_frac_piv,
            true_row.low_decile_frac,
            true_row.w_piv.mean / true_row.w_cand.mean
        ));
    }
    (pass, notes)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        bail!("usage: adaptive_rank_probe NUF1_DUMP OUTPUT_PREFIX");
    }
    let dump_path = PathBuf::from(&args[1]);
    let prefix = PathBuf::from(&args[2]);
    let total = Instant::now();
    eprintln!(
        "ml0_adaptive_rank_probe: dump={} prefix={}",
        dump_path.display(),
        prefix.display()
    );
    eprintln!(
        "scope: adaptive weighted selection only; no all-q Weinert / Umklapp / M-L completion claim"
    );

    let dump = read_adaptive(&dump_path)?;
    let n_mt = dump.n_mt();
    let n_int = dump.npts - n_mt;
    if dump.nk != 27
        || dump.nspin != 2
        || dump.bando != 13
        || dump.nband != 16
        || dump.npts != 27940
    {
        bail!(
            "unexpected schema nk={} nspin={} bando={} nband={} npts={}",
            dump.nk,
            dump.nspin,
            dump.bando,
            dump.nband,
            dump.npts
        );
    }
    if dump.kpt.len() != dump.nk {
        bail!("kpt length {} != nk {}", dump.kpt.len(), dump.nk);
    }
    if dump.nrad != 96 || dump.nang != 194 || dump.ninter != 24 {
        bail!(
            "unexpected grid nrad={} nang={} ninter={}",
            dump.nrad,
            dump.nang,
            dump.ninter
        );
    }
    let weight_sum: f64 = dump.quadrature.iter().copied().sum();
    let mut occ_sums = vec![0.0; dump.nspin];
    for spin in 0..dump.nspin {
        for k in 0..dump.nk {
            for band in 0..dump.bando {
                occ_sums[spin] += dump.occupation(spin, k, band);
            }
        }
    }
    let a1 = [dump.lat[0], dump.lat[1], dump.lat[2]];
    let lattice_constant = a1.iter().map(|c| c * c).sum::<f64>().sqrt();
    let mesh = KMesh::gamma_centred([3, 3, 3], lattice_constant)?;
    if mesh.len() != dump.nk {
        bail!("KMesh len {} != dump.nk {}", mesh.len(), dump.nk);
    }
    let layout = PairColumnLayout::new(dump.nk, dump.nband, None);
    eprintln!(
        "schema: NUF1 v2 nk={} nspin={} bando={} nband={} npts={} nrad={} nang={} ninter={} vol={:.12} weight_sum={:.12} n_mt={} n_int={} |a1|={:.9} layout n_columns={} encode(1,2,3)={} occ_sums={occ_sums:?}",
        dump.nk,
        dump.nspin,
        dump.bando,
        dump.nband,
        dump.npts,
        dump.nrad,
        dump.nang,
        dump.ninter,
        dump.vol,
        weight_sum,
        n_mt,
        n_int,
        lattice_constant,
        layout.n_columns()?,
        layout.encode(1, 2, 3)
    );

    let mut rows = Vec::new();
    for &m in &M_VALUES {
        let candidates = candidate_subset(n_mt, n_int, m)?;
        let true_weights: Vec<f64> = candidates.iter().map(|&p| dump.quadrature[p]).collect();
        validate_quadrature_weights(&true_weights)?;
        let uniform_value = true_weights.iter().sum::<f64>() / m as f64;
        let uniform_weights = vec![uniform_value; m];
        eprintln!(
            "M={m}: n_mt_cand={} n_int_cand={} true_w_sum={:.12} uniform_w={}",
            candidates.iter().filter(|&&p| p < n_mt).count(),
            candidates.iter().filter(|&&p| p >= n_mt).count(),
            true_weights.iter().sum::<f64>(),
            uniform_value
        );
        rows.extend(run_arm(
            &dump,
            &mesh,
            m,
            &candidates,
            &true_weights,
            "true",
            &true_weights,
            n_mt,
        )?);
        rows.extend(run_arm(
            &dump,
            &mesh,
            m,
            &candidates,
            &true_weights,
            "uniform",
            &uniform_weights,
            n_mt,
        )?);
    }

    let csv_path = PathBuf::from(format!("{}_ranks.csv", prefix.display()));
    write_csv(&csv_path, &rows)?;
    let (pass, notes) = gate_verdict(&rows);
    println!("csv={}", csv_path.display());
    println!(
        "baseline_m512_uniform_3e-3_recorded={}",
        BASELINE_M512_UNIFORM_RANK_3E3
    );
    if let Some(uni512) = rows
        .iter()
        .find(|r| r.m == 512 && r.arm == "uniform" && (r.threshold - 3.0e-3).abs() < 1.0e-15)
    {
        println!(
            "m512_uniform_3e-3_rank={}  delta_vs_recorded={:+}  (different candidate geometry, semantic 27x16x16 q=0 L+S columns, n_keep={}, adaptive dump vs regular 24^3 12x12 packed scan)",
            uni512.rank,
            uni512.rank as i64 - BASELINE_M512_UNIFORM_RANK_3E3 as i64,
            uni512.n_keep
        );
    }
    for row in &rows {
        println!(
            "M={} arm={} thresh={:.0e} rank={} n_keep={} mt_piv={}/{} ({:.3}) w_piv_mean={:.6e} w_cand_mean={:.6e} low_decile={:.3} sat={} wall={:.3}s",
            row.m,
            row.arm,
            row.threshold,
            row.rank,
            row.n_keep,
            row.n_mt_piv,
            row.n_mt_piv + row.n_int_piv,
            row.mt_frac_piv,
            row.w_piv.mean,
            row.w_cand.mean,
            row.low_decile_frac,
            row.saturated,
            row.wall_s
        );
    }
    for note in &notes {
        println!("gate: {note}");
    }
    println!(
        "verdict={} total_wall_s={:.3} peak_rss_kb={:?}",
        if pass { "PASS" } else { "FAIL" },
        total.elapsed().as_secs_f64(),
        peak_rss_kb()
    );
    if !pass {
        bail!("adaptive-weight rank gate failed");
    }
    Ok(())
}
