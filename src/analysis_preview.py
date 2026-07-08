from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass
class PreviewSaveState:
    save_dir: Path | None
    limit: int
    saved: int = 0

    @property
    def enabled(self) -> bool:
        return self.save_dir is not None

    @property
    def can_save(self) -> bool:
        return self.enabled and self.saved < self.limit


def preview_frame_name(analyzed: int, elapsed: float) -> str:
    return f"frame_{analyzed:05d}_{elapsed:.1f}s.jpg"


def maybe_save_preview(preview: Any, state: PreviewSaveState, analyzed: int, elapsed: float) -> PreviewSaveState:
    if state.can_save and state.save_dir is not None:
        cv2.imwrite(str(state.save_dir / preview_frame_name(analyzed, elapsed)), preview)
        state.saved += 1
    return state
