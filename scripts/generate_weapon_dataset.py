from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.weapon_training import background_images, generate_synthetic_dataset, icon_labels


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic weapon classifier images from icon assets.")
    parser.add_argument("--icons", default="main_icons", help="Directory containing one PNG per weapon class.")
    parser.add_argument("--backgrounds", default="sample", help="Directory containing background images.")
    parser.add_argument("--output-dir", default="outputs/generated_weapon_dataset", help="Output ImageFolder dataset root.")
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-labels", help="Optional label list path to write from icon class order.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generation plan without writing images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    icon_dir = project_path(args.icons)
    background_dir = project_path(args.backgrounds)
    output_dir = project_path(args.output_dir)
    labels_path = project_path(args.write_labels) if args.write_labels else None
    labels = icon_labels(icon_dir)
    backgrounds = background_images(background_dir)
    plan = {
        "icons": str(icon_dir),
        "backgrounds": str(background_dir),
        "output": str(output_dir),
        "classes": len(labels),
        "background_count": len(backgrounds),
        "images_per_class": args.images_per_class,
        "planned_images": len(labels) * args.images_per_class,
        "labels": str(labels_path) if labels_path else None,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("status: dry run only; no images written")
        return 0
    result = generate_synthetic_dataset(
        icon_dir,
        background_dir,
        output_dir,
        args.images_per_class,
        args.seed,
        labels_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
