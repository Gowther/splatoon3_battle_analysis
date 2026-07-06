from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms

from src.detection import crop_result, detections, player_lamps


class ImageTransform:
    def __init__(self) -> None:
        self.data_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.data_transform(img)


def load_weapon_names(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def weapon_model_output_count(model: nn.Module) -> Optional[int]:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return None


def classify_weapons(
    results,
    weapon_model,
    weapon_names: Sequence[str],
    device: str,
    transform: ImageTransform,
    ids: Dict[str, int],
) -> Optional[List[str]]:
    lamps = player_lamps(detections(results), ids)
    if len(lamps) != 8:
        return None

    output: List[str] = []
    with torch.no_grad():
        for lamp in lamps:
            crop = crop_result(results, lamp)
            if crop.size == 0:
                return None
            inputs = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
            logits = weapon_model(inputs)
            probs = F.softmax(logits, dim=1)
            index = int(torch.argmax(probs, dim=1).item())
            if index >= len(weapon_names):
                return None
            output.append(weapon_names[index])
    return output


def vote_weapons(samples: Iterable[Optional[List[str]]]) -> Optional[List[str]]:
    valid = [sample for sample in samples if sample and len(sample) == 8]
    if not valid:
        return None
    voted: List[str] = []
    for i in range(8):
        column = [sample[i] for sample in valid]
        voted.append(statistics.mode(column))
    return voted
