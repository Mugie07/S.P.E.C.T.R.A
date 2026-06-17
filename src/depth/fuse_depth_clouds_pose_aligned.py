from __future__ import annotations

from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fusion.depth_fusion import main


if __name__ == "__main__":
    main()
