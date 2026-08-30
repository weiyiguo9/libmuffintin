use muffintin_dft::ScfConfig;
use thiserror::Error;

use crate::InputError;
use crate::input::Task;
use crate::runner::{PreparedWorkflow, scf_config};

/// Failure to select and map exactly one DFT SCF task.
#[derive(Debug, Error)]
pub enum SingleDftScfConfigError {
    #[error("expected exactly one dft-scf task in the prepared workflow, found {count}")]
    TaskCount { count: usize },
    #[error(transparent)]
    Input(#[from] InputError),
}

/// Select and map the sole DFT SCF task in a prepared Input V3 workflow.
///
/// This is the single-SCF selection surface from Input V3 to frozen product
/// inputs. Selection follows [`PreparedWorkflow::tasks`] stable order and
/// rejects workflows containing zero or multiple DFT SCF tasks.
pub fn single_dft_scf_config(
    workflow: &PreparedWorkflow,
) -> Result<ScfConfig, SingleDftScfConfigError> {
    let mut selected = None;
    let mut count = 0;
    for task in &workflow.tasks {
        if matches!(&task.task, Task::DftScf { .. }) {
            count += 1;
            selected.get_or_insert(task);
        }
    }

    let task = selected
        .filter(|_| count == 1)
        .ok_or(SingleDftScfConfigError::TaskCount { count })?;
    scf_config(task, &workflow.checkpoint).map_err(Into::into)
}
