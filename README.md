# Tactile Manipulation Benchmark — Quick Start

A bimanual OpenArm with four tactile fingers in Isaac Sim, scripted
task demonstrations, LeRobot v3 datasets carrying the tactile channels, and
policy rollout in the same environment.

---

## 1. Requirements

| | version |
| --- | --- |
| GPU | NVIDIA, ≥ 10 GB VRAM |
| disk | ~25 GB free (Isaac Sim is ~18 GB) |
| driver | ≥ 580 |
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | v2.3.2 (`isaaclab` 0.54.2) + `isaaclab_contrib` (TacSL) |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| lerobot | 0.5.1, from source — optional, see below (dataset format v3.0) |

## 2. Install

Stage 0 below: `bash scripts/install_benchmark.sh`.

### 2.1 NVIDIA driver

```bash
sudo add-apt-repository universe && sudo apt update
sudo apt install -y nvidia-driver-580-server
sudo reboot
```

### 2.2 conda

```bash
sudo apt install curl
curl -fL# -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash && exec bash
```

### 2.3 the repo, then stage 0

```bash
sudo apt install -y git
git clone git@github.com:AIRSEAI/AIRSEAI-2.0.git ~/AIRSEAI-2.0
bash ~/AIRSEAI-2.0/scripts/install_benchmark.sh
```

Contributors: enable the DCO sign-off hook once per clone, see
[section 8](#8-contributing).

### 2.4 the robot asset

The robot asset comes from enactic's
[openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab). The
installer fetches it and converts it into the benchmark robot — the four
GelSight fingers and the wrist/head cameras welded on — cached at
`assets/usds/openarm_bimanual/openarm_tactile.usd`. Rebuild after changing
finger or camera placement: `python benchmark/run/build_asset.py`.
`--robot_usd` on any entry point uses a different asset.

## 3. The environments

Seven registered scenes (`ENVS` in `benchmark/run/launch_env.py`):

| `--env` | what it is |
| --- | --- |
| `minimal` | the robot and its four tactile fingers, no furniture. The reference scene for anything that looks like a sensing bug |
| `stack_cube` | pick up a cube and stack it on a second one. The template every task layer copies |
| `blind_peg_insertion` | pick up a peg, find a close-fitting hole by touch under an injected pose error, insert it |
| `blind_peg_dim` | `blind_peg_insertion` with the lights at 0.15× — same board, peg, expert and criterion; only the illumination changes |
| `sleeve_on_shaft` | slide a split sleeve onto a shaft, gripping hard enough to carry it but not so hard its bore closes |
| `texture_sort` | tell two similar-looking balls apart by surface and sort them, in light too dim to see the texture |
| `handover` | pass a bar from one arm to the other, releasing only once the receiver has the load |

## 4. The pipeline

Six stages.

```
0 install ──► 1 preview ──► 2 collect ──► 3 convert ──► 4 train ──► 5 evaluate
```

### Stage 0 — install

```bash
bash scripts/install_benchmark.sh
# no prompts: take every default
AIRSEAI_USE_DEFAULT=1 bash scripts/install_benchmark.sh
```

| | |
| --- | --- |
| prompts | conda env name, conda root, repo dir — each with a default; Enter accepts |
| `AIRSEAI_USE_DEFAULT=1` | skip the prompts |
| `AIRSEAI_ENV_NAME=` | conda env name, default `airseai` |
| `AIRSEAI_CONDA_ROOT=` | conda install, default `conda info --base` |
| `AIRSEAI_OUT_BASE=` | work root for logs / checkpoints / eval, default `~/airseai-benchmark` |
| `AIRSEAI_GIT_MIRROR=` | clone mirror prefix, used only after github.com fails |

Activate the env before every stage below:

```bash
source ~/miniconda3/bin/activate
conda activate airseai
```

### Stage 1 — preview an environment

```bash
# Isaac Sim GUI
python benchmark/run/launch_env.py --env <task> --gui
# headless: render a clip
python benchmark/run/launch_env.py --env <task> --video ~/<task>_clip.mp4
# headless: settle the scene and save one still
python benchmark/run/launch_env.py --env <task> --snapshot ~/<task>_snap.png
```

| | |
| --- | --- |
| `<task>` | one of the `--env` names in section 3 |
| `--gui` | interactive viewer |
| `--video PATH` | 300 steps at 30 fps by default (`--steps`, `--fps`) |
| `--snapshot PATH` | one still after settling |

### Stage 2 — collect demonstrations

```bash
bash scripts/run_collect.sh <task> <episodes> [dataset_out]
# RENDER=1 also writes the verification video; PREVIEWS=1 the tactile preview streams
RENDER=1 PREVIEWS=1 bash scripts/run_collect.sh <task> <episodes>
```

| | |
| --- | --- |
| `<task>` | `stack_cube` \| `blind_peg_insertion` \| `blind_peg_dim` \| `texture_sort` \| `sleeve_on_shaft` \| `handover` \| `handover_cup_transfer` |
| `<episodes>` | demonstrations to record |
| `[dataset_out]` | default `~/<task>_dataset` |
| `RENDER=1` | write `<dataset_out>_run/<task>.mp4` |
| `PREVIEWS=1` | add the human-viewable tactile streams to the dataset |

Output: the dataset at `dataset_out`, and `<dataset_out>_run/` with the
per-episode report. Task knobs (light level, clearance) are fixed by the task.
`dataset_out/review/third_person/` holds a third-person view of each kept
episode; it is not used in training by default.

### Stage 3 — convert to training datasets

```bash
bash scripts/run_convert.sh <raw_dataset_dir>
# a variant that already exists is skipped; FORCE=1 redoes it
FORCE=1 bash scripts/run_convert.sh <raw_dataset_dir>
```

| | |
| --- | --- |
| `<raw_dataset_dir>` | the dataset stage 2 wrote |
| `VARIANTS=` | subset of `vision tacff tacrgb`, default all three |
| `FORCE=1` | reconvert an existing variant |

Output: `~/airseai-benchmark/datasets/<name>_<variant>` for each variant, where
`<name>` is the raw dataset's directory name. Each has a third-person view of
its episodes under `review/third_person/`, not used in training by default.
The three differ only in which observation keys they carry:

| variant | observation keys |
| --- | --- |
| `vision` | `images.base`, `images.{right,left}_wrist`, `state(16)` |
| `tacff` | + `images.tacff_{right,left}_{middle,index}` — force field as a 3-channel image (R = depth, G/B = shear x/y) |
| `tacrgb` | + `images.tacrgb_{right,left}_{middle,index}` — raw Taxim tactile RGB |

### Stage 4 — train

```bash
DATASET=<name> VARIANT=tacff bash scripts/run_train.sh
```

| | |
| --- | --- |
| `DATASET=` | the raw dataset's directory name (what stage 3 named its outputs after) |
| `VARIANT=` | `vision` \| `tacff` \| `tacrgb` — which stage-3 dataset |
| `STEPS=` | default 50000 |
| `SEED=` | default 1000 |
| `GPU=` | default: the GPU with the most free memory |

Output: `~/airseai-benchmark/outputs/<name>_<variant>_act_s<seed>/checkpoints/`.
If the stage-3 dataset does not exist, training does not start — it prints
the `run_convert.sh` command to run and exits. The policy is ACT, which uses
ResNet18 for every image stream, tactile included.

### Stage 5 — evaluate

```bash
bash scripts/run_eval.sh <task> <checkpoint_dir> [episodes] [seed]
# VIDEO=1 also records base / wrist / third-person mp4s
VIDEO=1 bash scripts/run_eval.sh <task> <checkpoint_dir> [episodes]
```

| | |
| --- | --- |
| `<task>` | `stack_cube` \| `blind_peg_insertion` \| `blind_peg_dim` \| `texture_sort` |
| `<checkpoint_dir>` | lerobot `pretrained_model` dir (has `config.json`) |
| `[episodes]` | default 25 |
| `[seed]` | default 11 |
| `VIDEO=1` | also write base / wrist / third-person mp4s |
| `OUT=` | report dir, default `~/<task>_eval` |

Report: `<OUT>/eval_report.json`. The observation keys are read from the
checkpoint; nothing else to set.

## 5. What the dataset contains

LeRobot **v3.0**, readable by stock `LeRobotDatasetMetadata`; the tactile
`data` features need `lerobot_tactile_ext` to decode.

| key | dtype | shape | what it is |
| --- | --- | --- | --- |
| `observation.images.base` / `.left_wrist` / `.right_wrist` | video | 480×640×3 | cameras |
| `observation.images.{sensor}_tactile` ×4 | video | 288×384×3 | Taxim tactile RGB render |
| `observation.depth.{sensor}_tactile` ×4 | data (uint16, ffv1) | 288×384×1 | gel penetration depth, mm |
| `observation.ff_shear.{sensor}_tactile` ×4 | data (uint16, ffv1) | 288×384×2 | TacSL shear force field (x, y), ×1e3 |
| `observation.state` / `action` | float32 | 16 | 7 joints (deg) + gripper (mm) per arm, right first |
| `is_success` / `is_intervention` / `control_mode` | int64 | 1 | |

`{sensor}` ∈ `{left,right}_hand_{index,middle}_finger`. The quantisation
profile is recorded in each `data` feature's `info`, so no out-of-band
constants are needed.

Stage 3 turns a recording into three training datasets — each a new,
separate dataset; the recording is never modified. Same columns as above;
the last column says which recorded key each one comes from.

**`vision`** — cameras and proprioception only:

| key | dtype | shape | from |
| --- | --- | --- | --- |
| `observation.images.base` / `.left_wrist` / `.right_wrist` | video | 480×640×3 | as recorded |
| `observation.state` / `action` | float32 | 16 | as recorded |

**`tacrgb`** — `vision` plus the tactile camera images:

| key | dtype | shape | from |
| --- | --- | --- | --- |
| `observation.images.tacrgb_{arm}_{finger}` ×4 | video | 288×384×3 | `observation.images.{sensor}_tactile`, decoded and re-encoded under the new key |

**`tacff`** — `vision` plus the force field packed as an image:

| key | dtype | shape | from |
| --- | --- | --- | --- |
| `observation.images.tacff_{arm}_{finger}` ×4 | video | 288×384×3 | R = `observation.depth` / 3 mm; G, B = `observation.ff_shear` x, y mapped from [−6, +6] onto [0, 1] (`tactile_codec.tacff`) |

`{arm}_{finger}` ∈ `{right,left}_{middle,index}`.

Reading the `data` streams back (sim and real use the same codec; only the
key names differ):

```python
import json
from pathlib import Path
from lerobot_tactile_ext import decode_data_frames
from tactile_codec import QuantProfile, dequantize

root = Path("~/stack_cube_dataset").expanduser()          # a stage-2 recording
info = json.load(open(root / "meta/info.json"))
key = "observation.depth.right_hand_index_finger_tactile"
spec = info["features"][key]

u16 = decode_data_frames(root / "videos" / key / "chunk-000/file-000.mkv",
                         original_channels=spec["info"]["original_channels"])
depth_mm = dequantize(u16, QuantProfile(**spec["info"]["quant"]))
```

## 6. Layout

```
src/sim/            simulation: tactile sensing, robot asset builder,
                    envs/ (minimal unit) and tasks/ (task scenes)
src/tactile_codec/  quantisation profiles (float mm <-> uint16)
src/lerobot_tactile_ext/  ffv1 codec + LeRobot v3 dataset writer
scripts/run_*.sh    the pipeline entry points (collect, convert, train, eval)
                    plus install_benchmark.sh
benchmark/run/      launch_env, collect_dataset, build_asset, render_previews,
                    make_*  (task assets, authored at install), verify_codec
benchmark/tasks/    run_expert.py, the shared expert runner
benchmark/policy/   BC pipeline: dataset conversion, policy server
benchmark/eval/     evaluate.py, the policy evaluator
.githooks/          shared git hooks (DCO sign-off)
```

## 7. Simulation vs. the real robot

The real robot (Daimon DM-Tac) records the same cameras, state and action,
and these tactile keys per sensor:

| key | dtype | shape | sim | real |
| --- | --- | --- | --- | --- |
| `observation.images.{sensor}_tactile` | video | 288×384×3 | Taxim render | camera |
| `observation.depth.{sensor}_tactile` | data | 288×384×1 | gel penetration, mm | gel penetration, mm |
| `observation.ff_shear.{sensor}_tactile` | data | 288×384×2 | TacSL shear **force** field (x, y), uncalibrated ×1e3 | — |
| `observation.shear.{sensor}_tactile` | data | 288×384×2 | — | shear **displacement** field (x, y), mm |
| `observation.deformation.{sensor}_tactile` | data | 288×384×2 | — | deformation field (x, y), mm |
| `observation.state.force_{sensor}` | float32 | 6 | — | resultant wrench |

Tactile RGB and depth line up. The tangential channel does not: the sim's
`ff_shear` is a force in TacSL's own units, the real `shear` is a
displacement in millimetres — different physical quantities, carrying
different names on purpose so that a policy or a converter cannot mistake
one for the other.

`PREVIEWS=1` (stage 2) adds `observation.images.{depth,ff_shear}_{sensor}`
renders beside the raw streams on sim data, as the real recorder does for
its own fields.

## 8. Contributing

Every commit must carry a Developer Certificate of Origin (DCO) sign-off,
or the pull request's DCO check fails. The repo ships a hook that adds it
for you. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

From then on every `git commit`, from the terminal or an editor, ends with

```
Signed-off-by: Your Name <you@example.com>
```

taken from `user.name` and `user.email`. That email must match the one on
your GitHub account.

Check it works:

```bash
git commit --allow-empty -m "test sign-off"
git log -1 --format=%B        # the last line is Signed-off-by
git reset --soft HEAD~1       # drop the test commit, keep your changes
```

The hook does not run on rebase or cherry-pick. To sign commits that are
already on your branch, rewrite them and force-push:

```bash
git rebase --signoff main
git push --force-with-lease
```
