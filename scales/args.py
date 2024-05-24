from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoggingArgs:
    train_loss: bool = True
    validation_loss: bool = True
    learning_rate: bool = True
    global_gradient_norm: bool = False
    """Logs every certain number of steps."""
    log_step: int = 5
    log_dir: str | Path | None = None

    def should_log(self) -> bool:
        if self.train_loss or self.validation_loss or self.learning_rate or self.global_gradient_norm:
            return True
        return False
