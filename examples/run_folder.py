from __future__ import annotations

import os
from pathlib import Path


def create_slurm_script(
    folder_path: str, partition: str, max_time: str, job_name: str, gpu_per_job: int, results_path: str
) -> Path:
    """Create a slurm script to run all the configs in the folder_path. The generated script will be saved in the
    results_path/submit.

    Note: This function will not submit the job itself, but will return the path to the script.

    Args:
        folder_path (str): The path to the folder containing the config files
        partition (str): The partition to run the job on
        max_time (str): The maximum time the job can run
        job_name (str): The name of the job
        gpu_per_job (int): The number of GPUs per job
        results_path (str): The path to the results folder

    """

    folder_path = Path(folder_path)
    results_path = Path(results_path)
    log_path = results_path / "logs"
    script_folder = results_path / "submit"
    results_folder = results_path / "results"
    config_paths = []

    python_path = "examples/run.py"
    data_root = "/work/dlclarge1/garibovs-scales_n_arp/scaling_all_the_way/data"
    # Get all the config files in the folder
    for file in folder_path.iterdir():
        if file.is_dir():
            continue
        if file.suffix == ".yaml":
            config_paths.append(str(file))

    # Create the array of config paths
    array_str = f"AR=({' '.join(config_paths)})"

    # Create the command to run the python script
    command = (
        f"poetry run python {python_path} ${{AR[$SLURM_ARRAY_TASK_ID]}} {results_folder} --data_root_path={data_root}"
    )

    log_path.mkdir(parents=True, exist_ok=True)
    script_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    slurm_script = []
    slurm_script.append("#!/bin/bash")
    slurm_script.append("#SBATCH -c 2")
    slurm_script.append(f"#SBATCH --job-name {job_name}")
    slurm_script.append(f"#SBATCH --partition {partition}")
    slurm_script.append(f"#SBATCH --gres=gpu:{gpu_per_job}")
    slurm_script.append(f"#SBATCH --time {max_time}")
    slurm_script.append(f"#SBATCH --output {log_path}/%x.%A.%a.%N.out")
    slurm_script.append(f"#SBATCH --error {log_path}/%x.%A.%a.%N.err")
    slurm_script.append("#SBATCH --mail-type=FAIL")
    slurm_script.append(f"#SBATCH --array 1-{len(config_paths)}")
    slurm_script.append("#SBATCH -D /work/dlclarge1/garibovs-scales_n_arp/")
    slurm_script.append("source ./activate_env.sh")
    slurm_script.append("cd scaling_all_the_way")
    slurm_script.append(array_str)
    slurm_script.append(command)

    # Number script names in the folder
    script_name = len([None for i in script_folder.iterdir() if "sh" in i.suffix])
    script_path = script_folder / f"{script_name}.sh"
    with script_path.open("w") as f:
        f.write("\n".join(slurm_script))
    return script_path


if __name__ == "__main__":
    folder_path = "/work/dlclarge1/garibovs-scales_n_arp/configs/neps_selected/run=1"
    partition = "mlhiwidlc_gpu-rtx2080"
    max_time = "0:12:00"
    job_name = "selected_neps_run1"
    gpu_per_job = 1
    results_path = "/work/dlclarge1/garibovs-scales_n_arp/results/neps_selected/run=1"
    path = create_slurm_script(folder_path, partition, max_time, job_name, gpu_per_job, results_path)
    # Submit the job
    os.system(f"sbatch {path}")
