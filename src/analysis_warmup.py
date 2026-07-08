from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.weapons import ImageTransform, classify_weapons, vote_weapons


@dataclass
class WeaponWarmupState:
    votes: List[Optional[List[str]]] = field(default_factory=list)
    final_weapons: Optional[List[str]] = None


def record_weapon_vote(
    state: WeaponWarmupState,
    vote: Optional[List[str]],
    warmup_frames: int,
) -> WeaponWarmupState:
    if state.final_weapons is not None or len(state.votes) >= warmup_frames:
        return state
    if vote:
        state.votes.append(vote)
    if len(state.votes) >= warmup_frames:
        state.final_weapons = vote_weapons(state.votes)
    return state


def update_weapon_warmup(
    results: Any,
    *,
    warmup_frames: int,
    detection_ids: Dict[str, int],
    weapon_model: Any,
    weapon_names: List[str],
    weapon_transform: ImageTransform,
    device: str,
    state: WeaponWarmupState,
    logger: Callable[[str], None] | None = print,
) -> WeaponWarmupState:
    if state.final_weapons is not None or len(state.votes) >= warmup_frames:
        return state
    vote = classify_weapons(results, weapon_model, weapon_names, device, weapon_transform, detection_ids)
    previous_votes = len(state.votes)
    record_weapon_vote(state, vote, warmup_frames)
    if vote and len(state.votes) > previous_votes and logger:
        logger(f"Weapon warmup frame {len(state.votes)}/{warmup_frames}: {vote}")
    if state.final_weapons is not None and logger:
        logger(f"Weapon warmup complete: {state.final_weapons}")
    return state
