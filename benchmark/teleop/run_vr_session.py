#!/usr/bin/env python3
"""Preview or execute a task-specific session using the shared VR recipe."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))

from sim.vr_session import main

if __name__ == '__main__':
    raise SystemExit(main())
