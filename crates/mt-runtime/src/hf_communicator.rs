//! Optional process-wide communicator for the Gamma valence HF exchange build.

#[cfg(feature = "mpi")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::{AsRaw, Communicator, CommunicatorCollectives};
use num_complex::Complex64;

#[cfg(feature = "mpi")]
static MPI_WORLD_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "mpi")]
static MPI_WORLD_RANK: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "mpi")]
static MPI_WORLD_SIZE: AtomicUsize = AtomicUsize::new(1);

/// Select the initialized MPI world communicator for Gamma valence HF.
///
/// The first production MPI driver is deliberately world-only. rsmpi owns
/// communicators constructed through `FromRaw`, so a borrowed
/// `MPI_COMM_WORLD` must instead be recovered with
/// [`SimpleCommunicator::world`].
#[cfg(feature = "mpi")]
pub fn set_hf_mpi_communicator(raw: mpi::ffi::MPI_Comm) {
    assert!(
        mpi::is_initialized(),
        "MPI must be initialized before setting the HF communicator"
    );
    let world = SimpleCommunicator::world();
    assert_eq!(
        raw,
        world.as_raw(),
        "Gamma valence HF currently accepts only MPI_COMM_WORLD"
    );
    MPI_WORLD_RANK.store(
        usize::try_from(world.rank()).expect("MPI rank must be nonnegative"),
        Ordering::Relaxed,
    );
    MPI_WORLD_SIZE.store(
        usize::try_from(world.size()).expect("MPI size must be nonnegative"),
        Ordering::Relaxed,
    );
    MPI_WORLD_ACTIVE.store(true, Ordering::Release);
}

#[cfg(feature = "mpi")]
fn active_world() -> Option<SimpleCommunicator> {
    (mpi::is_initialized() && MPI_WORLD_ACTIVE.load(Ordering::Acquire))
        .then(SimpleCommunicator::world)
}

pub(crate) fn rank_and_size() -> (usize, usize) {
    #[cfg(feature = "mpi")]
    if MPI_WORLD_ACTIVE.load(Ordering::Acquire) {
        return (
            MPI_WORLD_RANK.load(Ordering::Relaxed),
            MPI_WORLD_SIZE.load(Ordering::Relaxed),
        );
    }
    (0, 1)
}

pub(crate) fn all_reduce_sum_complex(_values: &mut [Complex64]) {
    #[cfg(feature = "mpi")]
    if let Some(world) = active_world() {
        let send = _values
            .iter()
            .flat_map(|value| [value.re, value.im])
            .collect::<Vec<_>>();
        let mut receive = vec![0.0; send.len()];
        world.all_reduce_into(&send[..], &mut receive[..], SystemOperation::sum());
        for (value, reduced) in _values.iter_mut().zip(receive.chunks_exact(2)) {
            *value = Complex64::new(reduced[0], reduced[1]);
        }
    }
}
