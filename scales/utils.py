from pathlib import Path
from typing import Tuple

import lightning as L
from litgpt.utils import save_config


def save_checkpoint(fabric: L.Fabric, state: dict, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving state to {str(checkpoint_dir)}")
    fabric.save(Path(checkpoint_dir / "lit_model.pth"), state)
    print(checkpoint_dir.parent)
    if fabric.global_rank == 0:
        save_config(state["model"].config, checkpoint_dir)


def load_checkpoint(fabric: L.Fabric, checkpoint_dir: str | Path) -> Tuple[dict, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    fabric.print(f"Loading state from {str(checkpoint_dir)}")
    state = fabric.load(path=Path(checkpoint_dir / "lit_model.pth"))
    model_config_path = Path(checkpoint_dir / "model_config.yaml")
    return state, model_config_path
