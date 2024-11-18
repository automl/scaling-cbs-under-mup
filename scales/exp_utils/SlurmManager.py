from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import subprocess
import os
import time

import yaml

def check_run_finished(run_folder: Path) -> bool:
    result_file = run_folder / "result.yaml"
    info_file = run_folder / "info.yaml"
    print(f"Checking {run_folder}")
    if run_folder.exists() and result_file.exists() and info_file.exists():
        with result_file.open("r", encoding="utf-8") as stream:
            res = yaml.safe_load(stream)
        with info_file.open("r", encoding="utf-8") as stream:
            _info = yaml.safe_load(stream)
        
        if "train_target" in _info and int(res["train_steps"]) >= int(_info["train_target"]):
            return True
    print(f"Run is not finished")
    return False

@dataclass
class SlurmManager:
    """
    2 main inputs:
        root_configs_path
        root_outputs_path

    General structure of the root_outputs_path becomes:
    root_outputs_path/
    ------------------results/
    --------------------------config_path.stem/ (for config_path in config_paths)
    ......................... OR
    --------------------------main output/ (single job submission case)
    ------------------submit/
    -------------------------j.sh (j in range(previous submissions))
    ------------------log/
    ----------------------%x.%A.%a.%N.out (for _ in config_paths)
    ----------------------%x.%A.%a.%N.err (for _ in config_paths)
    """
    # TODO: Refactor this
    # TODO: Refactor to be at least somewhat consistent
    experiment_name: str = "slurm"
    manager_log_folder: Path | str | None = field(default=None)
    root_configs_path: Path | str | None = field(default=None, init=False)
    root_output_path: Path | str | None = field(default=None, init=False)
    config_paths: list[str] = field(default_factory=list, init=False)
    python_path: str | None = field(default=None, init=False)
    data_root: str | None = field(default=None, init=False)
    __script_path: Path | None = field(default=None, init=False)

    # @property
    def results_folder(self, root_output_path: Path | None = None) -> Path:
        root_output_path = root_output_path or self.root_output_path
        return root_output_path / "results"

    # @property
    def script_folder(self, root_output_path: Path | None = None):
        root_output_path = root_output_path or self.root_output_path
        return root_output_path / "submit"

    # @property
    def log_folder(self, root_output_path: Path | None = None):
        root_output_path = root_output_path or self.root_output_path
        return root_output_path / "log"

    @property
    def script_path(self):
        # Number script names in the folder
        if self.__script_path is not None:
            return self.__script_path
        script_name = len([None for i in self.script_folder().iterdir() if "sh" in i.suffix])
        script_path = self.script_folder() / f"{script_name}.sh"
        self.__script_path = script_path
        return self.__script_path

    @property
    def command(self):
        if self.root_configs_path:
            return (f"srun poetry run python {self.python_path} ${{AR[$SLURM_ARRAY_TASK_ID]}}"
                    f" {self.results_folder()} --data_root_path={self.data_root}")
        elif self._command is None:
            return f"""srun poetry run python -c "from scales.neps_utils import dynamic_micro_batch; dynamic_micro_batch(r'{self.root_output_path.resolve()}')" """
        else:
            return self._command

    @property
    def list_results_paths(self):
        """Returns list of paths for runs that has already started"""
        return [_dir for _dir in self.results_folder().iterdir() if _dir.is_dir()]

    def __post_init__(self):
        self.manager_log_folder = Path(self.manager_log_folder) if self.manager_log_folder is not None else None
        self.root_configs_path = None
        self.root_output_path = None
        self.python_path = "examples/run.py" if self.python_path is None else self.python_path
        self.data_root = "/work/dlclarge1/garibovs-scales_n_arp/scaling_all_the_way/data"
        self.job_id = ""
        self._command = None
        self.__previous_last_log_line = None

    def slurm_script_boilerplate(self,
                                 partition,
                                 max_time: str,
                                 cpu_per_job: int = 1,
                                 gpu_per_job: int = 1,
                                 activate_env: bool = False,
                                 array_throttle: int | None = 10,
                                 ):
        slurm_script = list()
        slurm_script.append("#!/bin/bash")
        slurm_script.append(f"#SBATCH -c {cpu_per_job}")
        slurm_script.append(f"#SBATCH --job-name {self.experiment_name}")
        slurm_script.append(f"#SBATCH --partition {partition}")
        slurm_script.append(f"#SBATCH --gres=gpu:{gpu_per_job}")
        slurm_script.append(f"#SBATCH --ntasks-per-node={gpu_per_job}")
        slurm_script.append(f"#SBATCH --time {max_time}")
        slurm_script.append(f"#SBATCH --output {self.log_folder()}/%x.%A.%a.%N.out")
        slurm_script.append(f"#SBATCH --error {self.log_folder()}/%x.%A.%a.%N.err")
        slurm_script.append("#SBATCH --mail-type=FAIL")
        if self.root_configs_path and array_throttle is not None:
            slurm_script.append(f"#SBATCH --array 0-{len(self.config_paths) - 1}%{array_throttle}")
        if self.cwd:
            slurm_script.append(f"#SBATCH -D {self.cwd}")
        if activate_env:
            slurm_script.append("source ./activate_env.sh")
        return slurm_script

    def create_slurm_script_single(self,
                                   config_path: str | Path,
                                   root_output_path: str | Path,
                                   partition: str,
                                   max_time: str,
                                   gpu_per_job: int,
                                   cwd: str | None = None,
                                   command: str | None = None,
                                   activate_env: bool = False):
        self.cwd = cwd
        self._command = command
        self.root_configs_path = None
        self.root_output_path = Path(root_output_path)
        self.log_folder().mkdir(parents=True, exist_ok=True)
        self.script_folder().mkdir(parents=True, exist_ok=True)
        self.results_folder().mkdir(parents=True, exist_ok=True)

        slurm_script = self.slurm_script_boilerplate(partition=partition,
                                      max_time=max_time,
                                      cpu_per_job=2,
                                      gpu_per_job=gpu_per_job,
                                      activate_env=activate_env,
                                      array_throttle=None,)
        slurm_script.append(self.command)

        with self.script_path.open("w") as f:
            f.write("\n".join(slurm_script))


    def get_log_files(self, root_output_path: str | Path | None = None) -> tuple[Path, Path]:
        """
        Not implemented for the array jobs
        :param root_output_path:
        :return:
        """
        log_dir = self.log_folder(Path(root_output_path))
        out_files = [file for file in log_dir.iterdir() if file.is_file() and file.suffix == ".out"]
        out_files.sort(key=lambda x: x.stat().st_mtime)
        err_files = [file for file in log_dir.iterdir() if file.is_file() and file.suffix == ".err"]
        err_files.sort(key=lambda x: x.stat().st_mtime)

        return out_files[-1], err_files[-1]

    def check_halt(self, result_path: str | Path | None = None):
        """
        Not implemented for the Array jobs
        :param result_path:
        :return:
        """
        out_log, _ = self.get_log_files(result_path)
        with out_log.open("r", encoding="utf-8") as log_file:
            # get the last two lines of the log file
            last_line = "".join(log_file.readlines()[-2:])
        if last_line != self.__previous_last_log_line:
            self.__previous_last_log_line = last_line
            return False
        else:
            return True


    def create_slurm_script_folder(self,
                                   partition: str,
                                   max_time: str,
                                   gpu_per_job: int,
                                   root_configs_path: Path | str | None,
                                   root_output_path: Path | str | None,
                                   cwd: str | None = "/work/dlclarge1/garibovs-scales_n_arp/",
                                   array_throttle: int = 12,
                                   activate_env: bool = True) -> Path:

        self.root_configs_path = Path(root_configs_path) if root_configs_path is not None else self.root_configs_path
        self.root_output_path = Path(root_output_path) if root_output_path is not None else self.root_output_path
        self.log_folder().mkdir(parents=True, exist_ok=True)
        self.script_folder().mkdir(parents=True, exist_ok=True)
        self.results_folder().mkdir(parents=True, exist_ok=True)
        self.config_paths = []
        self.cwd = cwd


        # Get all the config files in the folder
        for file in self.root_configs_path.iterdir():
            if file.is_dir():
                continue
            if file.suffix == ".yaml":
                self.config_paths.append(str(file))

        # Create the array of config paths
        array_str = f"AR=({' '.join(self.config_paths)})"

        slurm_script = self.slurm_script_boilerplate(partition=partition,
                                                     max_time=max_time,
                                                     cpu_per_job=2,
                                                     gpu_per_job=gpu_per_job,
                                                     activate_env=activate_env,
                                                     array_throttle=array_throttle, )

        slurm_script.append(array_str)
        slurm_script.append("cd scaling_all_the_way")
        slurm_script.append(self.command)

        with self.script_path.open("w") as f:
            f.write("\n".join(slurm_script))

    def check_finished(self):
        """
        Note: only considers jobs that has already started evaluating
        To check if all submitted jobs are finished compaer the number of finished jobs agains submitted ones
        :return: how many jobs have finished
        """
        if self.root_configs_path:
            completed = [False] * len(self.list_results_paths)
            for idx, path in enumerate(self.list_results_paths):
                path = Path(path)
                completed[idx] = check_run_finished(path)
            return completed
        else:
            return [check_run_finished(self.results_folder())]

    def check_run_completed(self):
        completed = self.check_finished()
        if self.root_configs_path:
            return sum(completed) == len(self.config_paths)
        else:
            return completed[0]


    def submit(self):
        print(f"Submitting {self.script_path}")
        submit = subprocess.run(["sbatch", str(self.script_path)], capture_output=True, text=True, input="y\n")
        # print(submit.stdout)
        self.job_id = submit.stdout.strip().split(" ")[-1].strip()
        print(f"Job ID: {self.job_id}")