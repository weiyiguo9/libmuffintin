//! Opt-in process-wide HF progress and wall-clock diagnostics.

use std::fmt::Arguments;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// Diagnostic output level. Library calls are quiet unless explicitly enabled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum HfVerbosity {
    Quiet = 0,
    Progress = 1,
    Timings = 2,
}

static VERBOSITY: AtomicU8 = AtomicU8::new(HfVerbosity::Quiet as u8);

/// Set the process-wide HF diagnostic level; output is written to stderr.
pub fn set_hf_verbosity(verbosity: HfVerbosity) {
    VERBOSITY.store(verbosity as u8, Ordering::Relaxed);
}

pub(crate) fn hf_progress(message: Arguments<'_>) {
    if VERBOSITY.load(Ordering::Relaxed) >= HfVerbosity::Progress as u8 {
        eprintln!("[hf] {message}");
    }
}

/// An end marker measures scope lifetime, including error exits, not success.
pub(crate) struct HfPhaseTimer {
    phase: &'static str,
    start: Option<Instant>,
}

impl HfPhaseTimer {
    pub(crate) fn new(phase: &'static str) -> Self {
        let start = (VERBOSITY.load(Ordering::Relaxed) >= HfVerbosity::Timings as u8).then(|| {
            eprintln!("[hf timing] begin {phase}");
            Instant::now()
        });
        Self { phase, start }
    }
}

impl Drop for HfPhaseTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            eprintln!(
                "[hf timing] end {} elapsed_s={:.6}",
                self.phase,
                start.elapsed().as_secs_f64()
            );
        }
    }
}
