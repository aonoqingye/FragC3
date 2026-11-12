"""Separate exact best-checkpoint selection from meaningful-improvement stopping."""

from dataclasses import asdict, dataclass
import math


@dataclass
class EarlyStopping:
    mode: str
    patience: int
    min_delta: float
    grace_after_lr_drop: int = 5
    best_score: float | None = None
    best_epoch: int = 0
    anchor_score: float | None = None
    anchor_epoch: int = 0
    bad_epochs: int = 0
    last_lr_drop_epoch: int = 0
    completed_epoch: int = 0

    def __post_init__(self):
        if self.mode not in ("min", "max") or self.patience < 1:
            raise ValueError("Invalid early-stopping mode or patience")
        if not math.isfinite(self.min_delta) or self.min_delta < 0 or self.grace_after_lr_drop < 0:
            raise ValueError("Invalid early-stopping delta or learning-rate grace")

    def better(self, score, reference, delta=0.0):
        if reference is None:
            return True
        return score > reference + delta if self.mode == "max" else score < reference - delta

    def observe(self, score, epoch, lr_before, lr_after):
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation selection score")
        if epoch != self.completed_epoch + 1:
            raise ValueError("Early-stopping epochs must be contiguous")
        if len(lr_before) != len(lr_after) or not lr_before:
            raise ValueError("Learning-rate group state mismatch")
        if any(not math.isfinite(x) or x <= 0 for x in list(lr_before) + list(lr_after)):
            raise ValueError("Invalid learning-rate state")
        best_improved = self.better(score, self.best_score)
        significant = self.better(score, self.anchor_score, self.min_delta)
        if best_improved:
            self.best_score, self.best_epoch = score, epoch
        if significant:
            self.anchor_score, self.anchor_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        reduced = any(new < old for old, new in zip(lr_before, lr_after))
        if reduced:
            self.last_lr_drop_epoch = epoch
        self.completed_epoch = epoch
        return {
            "best_improved": best_improved,
            "significant_improvement": significant,
            "lr_reduced": reduced,
            "early_stop_ready": self.should_stop(),
        }

    def should_stop(self):
        eligible = self.bad_epochs >= self.patience
        # Only an actual rate reduction starts grace. Reaching min_lr without
        # changing a rate must not repeatedly postpone stopping.
        grace_complete = not self.last_lr_drop_epoch or (
            self.completed_epoch - self.last_lr_drop_epoch >= self.grace_after_lr_drop
        )
        return eligible and grace_complete

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        restored = type(self)(**state)
        for field in ("mode", "patience", "min_delta", "grace_after_lr_drop"):
            if getattr(restored, field) != getattr(self, field):
                raise ValueError("Early-stopping policy changed on resume: " + field)
        if (
            restored.completed_epoch < 1
            or restored.bad_epochs < 0
            or not 1 <= restored.best_epoch <= restored.completed_epoch
            or not 1 <= restored.anchor_epoch <= restored.completed_epoch
            or not 0 <= restored.last_lr_drop_epoch <= restored.completed_epoch
            or restored.best_score is None
            or restored.anchor_score is None
            or not math.isfinite(restored.best_score)
            or not math.isfinite(restored.anchor_score)
        ):
            raise ValueError("Invalid saved early-stopping state")
        if restored.bad_epochs != restored.completed_epoch - restored.anchor_epoch:
            raise ValueError("Saved early-stopping counter and anchor disagree")
        self.__dict__.update(restored.__dict__)
