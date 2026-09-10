"""Validate a contribution scene without launching an interactive viewer."""

from __future__ import annotations

import argparse

from ibero.core.scene_loader import SceneLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="path to scenes/<scene-id>")
    args = parser.parse_args()
    scene = SceneLoader().validate(args.scene)
    print(f"Scene '{scene.config['id']}' is valid")
    print(f"scene_hash={scene.scene_hash}")


if __name__ == "__main__":
    main()
