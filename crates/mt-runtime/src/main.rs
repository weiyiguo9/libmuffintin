use std::env;
use std::process::ExitCode;

use muffintin::{
    PreparedWorkflow, CheckpointPhysics, TaskResult, WorkflowResult, execute_prepared_with,
    load_input_path,
};

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| "muffintin".into())
        .to_string_lossy()
        .into_owned();
    let Some(input_path) = arguments.next() else {
        eprintln!("usage: {program} <input.toml>");
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {program} <input.toml>");
        return ExitCode::from(2);
    }

    let prepared = match load_input_path(input_path) {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!("muffintin: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut physics = match CheckpointPhysics::new(&prepared.checkpoint) {
        Ok(physics) => physics,
        Err(error) => {
            eprintln!("muffintin: could not construct checkpoint DFT kernel: {error}");
            return ExitCode::FAILURE;
        }
    };
    match execute_prepared_with(&prepared, &mut physics) {
        Ok(result) => {
            print_summary(&prepared, &result);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("muffintin: {error}");
            ExitCode::FAILURE
        }
    }
}

fn print_summary(workflow: &PreparedWorkflow, result: &WorkflowResult) {
    for (task, output) in workflow.tasks.iter().zip(&result.tasks) {
        match output {
            TaskResult::Scf(state) => println!(
                "task {} scf iterations={} total_energy_ha={:.16e}",
                task.id,
                state.iterations(),
                state.energy.total.get()
            ),
            TaskResult::Bands(bands) => {
                println!("task {} bands points={}", task.id, bands.points.len());
                for point in &bands.points {
                    let energies = point
                        .energies
                        .iter()
                        .map(|energy| format!("{:.16e}", energy.get()))
                        .collect::<Vec<_>>()
                        .join(",");
                    println!(
                        "  {} k=[{:.16e},{:.16e},{:.16e}] energies_ha=[{}]",
                        point.label, point.k[0], point.k[1], point.k[2], energies
                    );
                }
            }
            TaskResult::Dos(dos) => {
                let tetrahedron = &dos.tetrahedron;
                let edges = tetrahedron
                    .edges
                    .iter()
                    .map(|edge| format!("{:.16e}", edge.get()))
                    .collect::<Vec<_>>()
                    .join(",");
                let density = tetrahedron
                    .density
                    .iter()
                    .map(|value| format!("{value:.16e}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let integrated = tetrahedron
                    .integrated_count
                    .iter()
                    .map(|value| format!("{value:.16e}"))
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "task {} dos edges_ha=[{}] density_per_ha=[{}] integrated_count=[{}]",
                    task.id, edges, density, integrated
                );
            }
        }
    }
}
