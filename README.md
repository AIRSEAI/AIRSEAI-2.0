# AIRSEAI 2.0

An open-source embodied-AI platform: a dual-arm
manipulator with vision-based tactile sensors, one data pipeline, and two
places it runs — sim and the real robot.

**In simulation, this repository.** A bimanual OpenArm with four tactile
fingers in Isaac Sim, a scripted expert per task, LeRobot v3 datasets
carrying the tactile channels, ACT training on them, and policy rollout in
the same scenes.

**On the real robot.** The same arms, the same recorded keys, collected by leader-follower teleoperation. Sim and real do not line up everywhere, and quantifying where they part is why the benchmark exists.

## The tasks

| | | |
| :---: | :---: | :---: |
| <img src="docs/media/stack_cube.webp" width="250" alt="stack_cube"> | <img src="docs/media/blind_peg_insertion.webp" width="250" alt="blind_peg_insertion"> | <img src="docs/media/sleeve_on_shaft.webp" width="250" alt="sleeve_on_shaft"> |
| `stack_cube` — stack one cube on another | `blind_peg_insertion` — insert a peg into a hole that is not where the robot is told | `sleeve_on_shaft` — slide a squeezable sleeve onto a shaft |
| <img src="docs/media/texture_sort.webp" width="250" alt="texture_sort"> | <img src="docs/media/handover.webp" width="250" alt="handover"> | <img src="docs/media/cap_unscrew.webp" width="250" alt="cap_unscrew"> |
| `texture_sort` — sort two balls that differ only in surface, in the dark | `handover` — hand a spring-loaded sleeve between the arms, from one cup to the other | `cap_unscrew` — hold a bottle, unscrew its cap and place it in a tray |

---

## 1. Requirements

| | version |
| --- | --- |
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | v2.3.2 (`isaaclab` 0.54.2) + `isaaclab_contrib` (TacSL) |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| lerobot | 0.5.1, compatible source installed in `airseai` (dataset format v3.0) |

See the [Isaac Sim 5.1 system requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
for minimum hardware requirements and supported drivers.

## 2. Robot asset

The robot asset comes from enactic's
[openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab). The
installer prepares the bimanual robot with four GelSight fingers and
wrist/head cameras at `assets/usds/openarm_bimanual/openarm_tactile.usd`.

## 3. The environments

The registered scenes include the task environments and `minimal`. The clips
at the top of this file show demonstrations of the task scenes:

| `--env` | what it is |
| --- | --- |
| `minimal` | the robot and its four tactile fingers, no furniture. The reference scene for anything that looks like a sensing bug |
| `stack_cube` | pick up a cube and stack it on a second one. The template every task layer copies |
| `blind_peg_insertion` | pick up a peg, find a close-fitting hole by touch under an injected pose error, insert it |
| `sleeve_on_shaft` | slide a split sleeve onto a shaft, gripping hard enough to carry it but not so hard its bore closes |
| `texture_sort` | tell two similar-looking balls apart by surface and sort them, in light too dim to see the texture |
| `handover` | pinch a spring-loaded split sleeve out of the orange cup with the left arm, hand it to the right arm, and seat it over the key in the teal cup, holding it open just enough to fit |
| `mug_on_hook` | carry a mug by its body and hang its handle on a peg |
| `cap_unscrew` | hold a bottle, unscrew its cap with the other hand, and place it in a tray |

## 4. The pipeline

```text
0 install → 1 preview → 2 collect → 3 convert → 4 train → 5 evaluate
```

### Stage 0 — install

#### NVIDIA driver

Install an NVIDIA driver compatible with your GPU model and the selected
Isaac Sim release. Follow the [Isaac Sim driver requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
Skip this step if a compatible driver is already installed.

#### Conda

Skip this step if conda is already installed.

```bash
sudo apt install -y curl
curl -fL -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
```

#### Install the airseai-2.0

```bash
sudo apt install -y git
git clone git@github.com:AIRSEAI/AIRSEAI-2.0.git ~/AIRSEAI-2.0
cd ~/AIRSEAI-2.0
bash scripts/install_airseai.sh
```

The installer asks for four settings. Press Enter to accept the displayed default.

| Prompt | Meaning | Default |
| --- | --- | --- |
| `conda env name` | Environment to create or reuse | `airseai` |
| `conda root` | Conda installation directory | Detected installation, falling back to `~/miniconda3` |
| `repo dir` | AIRSEAI repository directory | Repository containing the installer |
| `output root` | Directory for generated files | `outputs/` under the selected `repo dir` |

#### Activate the installed environment

After installation completes successfully, activate the environment in the
terminal used for the remaining stages:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate airseai
```

The selected output root is saved as `OUTPUT_ROOT` in this conda environment
and loaded automatically on activation.

### Stage 1 — preview an environment

Task environments use the shared Isaac Lab ManagerBased environment.
`--num_envs` defaults to 1. Larger values create independent copies of the
selected task in one Isaac Sim scene and process, advancing together at each
simulation step.

```bash
# Open an interactive view of four environments
python benchmark/run/launch_env.py --env stack_cube --num_envs 4 --gui
# Save a 30-second grid video of four environments
python benchmark/run/launch_env.py --env stack_cube --num_envs 4 \
    --video "$OUTPUT_ROOT/preview/stack_cube.mp4" --steps 1800
# Save a snapshot of four environments
python benchmark/run/launch_env.py --env stack_cube --num_envs 4 \
    --snapshot "$OUTPUT_ROOT/preview/stack_cube.png"
```

### Stage 2 — collect demonstrations

#### Scripted expert

```bash
# Run 50 attempts in one environment, saving review and tactile preview videos
RENDER=1 bash scripts/run_collect.sh stack_cube 50 \
    "$OUTPUT_ROOT/raw/stack_cube_expert"
# Run 200 total attempts across four environments with the same video outputs
NUM_ENVS=4 RENDER=1 \
    bash scripts/run_collect.sh stack_cube 200 \
    "$OUTPUT_ROOT/raw/stack_cube_expert_parallel"
```

The requested count is the total number of attempts, not successful episodes,
for both single- and multi-environment collection. By default, failed trajectories
are excluded from the training dataset and are not replaced by extra attempts.
For example, 50 attempts with 42 successes produce a dataset of 42 episodes.
Failed-attempt diagnostics remain under `<dataset>/run/attempts/`.

When `INIT_STATES` is supplied, each state is attempted once; the number of states
overrides the requested episode count. Without it, each attempt draws a new state.
The Python entry point's `--keep_unclean` option explicitly allows unsuccessful
trajectories to be retained; it is disabled by default.

| Option | Meaning / default |
| --- | --- |
| `NUM_ENVS` | Concurrent environments; `1` |
| `GPUS` | One GPU ID for native vector tasks; `0` |
| `SEED` | Collection seed; `7` |
| `INIT_STATES` | Optional initial-state JSON; absent means independent draws |
| `RENDER=1` | Record review videos |
| `PREVIEWS=1` | Add depth heatmaps and shear-field arrow videos for visual inspection |

#### Keyboard and VR teleoperation (Todo)

```bash
python benchmark/teleop/run_teleop.py --task stack_cube \
    --teleop_device keyboard --episodes 10 \
    --dataset "$OUTPUT_ROOT/raw/stack_cube_teleop"
```

Omit `--dataset` to practise without recording. `--episodes` counts attempts;
the report records manual success marks and task criterion results separately.
Native teleoperation accepts `--num_envs 1` only and supports `--init_states`.
Press SPACE to begin/pause, ENTER to mark success and save, BACKSPACE to discard,
R to discard and reset to the next start, TAB to switch arms, and K to toggle jaws.
Pausing freezes physics so no unrecorded interval enters a demonstration.

Use `--teleop_device vr --xr` for Isaac Lab OpenXR hand tracking. The selected
hand drives the selected arm with relative pose increments; keyboard controls
manage the jaws and recording. Tracking loss freezes physics and reacquisition
reanchors hand motion. This requires a compatible OpenXR runtime and tracked
hands; the operator/device workflow has not yet been validated.

| Output | Path |
| --- | --- |
| Raw dataset | `$OUTPUT_ROOT/raw/stack_cube_teleop/` |
| Episode report | `$OUTPUT_ROOT/raw/stack_cube_teleop_run/stack_cube_teleop.json` |
| Review videos | `$OUTPUT_ROOT/raw/stack_cube_teleop/review/third_person/` |

### Stage 3 — convert and combine datasets

Combine multiple demonstration datasets from stage 2:

```bash
DATASET=merged_demonstrations bash scripts/run_convert.sh \
    "$OUTPUT_ROOT/raw/stack_cube_1" \
    "$OUTPUT_ROOT/raw/stack_cube_2"
```

One or more input directories are accepted;
`DATASET` names the combined output. Source recordings remain unchanged.
Inputs must have compatible task, action, and observation definitions.

| Option | Meaning / default |
| --- | --- |
| `DATASET` | Output dataset name; required for multiple inputs, otherwise the input directory name |
| `VARIANTS` | Space-separated variants; `vision tacff tacrgb` |
| `FORCE=1` | Regenerate existing outputs; otherwise skip them |

| Variant | Inputs | Combined output |
| --- | --- | --- |
| `vision` | Cameras and robot state | `$OUTPUT_ROOT/datasets/merged_demonstrations_vision/` |
| `tacff` | Vision plus force-field images | `$OUTPUT_ROOT/datasets/merged_demonstrations_tacff/` |
| `tacrgb` | Vision plus tactile RGB images | `$OUTPUT_ROOT/datasets/merged_demonstrations_tacrgb/` |

All variants contain the same combined episodes and actions. Each output
includes `meta/source_episodes.json` mapping episodes to their source
recording, and review videos under `review/third_person/`.

### Stage 4 — train a policy

```bash
DATASET=merged_demonstrations VARIANT=tacff POLICY=act GPU=0 \
    STEPS=50000 SEED=1000 bash scripts/run_train.sh
```

| Option | Meaning / default |
| --- | --- |
| `DATASET` | Dataset name from stage 3 |
| `VARIANT` | `vision`, `tacff`, or `tacrgb` |
| `POLICY` | LeRobot policy type; `act` |
| `GPU` | Training GPU; defaults to the GPU with the most free memory |
| `STEPS` | Training steps; `50000` |
| `SEED` | Training seed; `1000` |
| `BATCH` / `WORKERS` | Batch size / loader workers; `8` / `8` |
| `SAVE_FREQ` / `LOG_FREQ` | Checkpoint / log interval; `10000` / `200` steps |
| `DS` | Explicit converted dataset path |
| `BASE` | Training output root; defaults to `OUTPUT_ROOT` |
| `DEFAULT_PY` | Python executable from the training environment |

Pass additional LeRobot options as shell arguments, for example
`bash scripts/run_train.sh --policy.chunk_size=100`.
Set `GPU` explicitly on shared servers according to the available allocation.
Relative `DS` and `BASE` paths are resolved from the calling directory.
For the 500-episode stack-cube recording:

```bash
DATASET=stack_cube_s51_n500 VARIANT=vision POLICY=act GPU=0 \
    DS="<converted_dataset_root>/stack_cube_s51_n500_vision" \
    BASE="<task_output_root>/stage4" STEPS=50000 SEED=1000 \
    bash scripts/run_train.sh
```

Repeat with `VARIANT=tacrgb` or `tacff` and the matching `DS` path.
Texture-sort training uses `example/texture_sort/train.sh`.
Evaluation initial states are not training inputs. Only launch after conversion
and dataset validation complete.

ACT is the initial baseline. Other model families may require additional
model configuration and dependencies.

- Checkpoints: `$BASE/outputs/${DATASET}_${VARIANT}_${POLICY}_s${SEED}/checkpoints/`.
- Log: `$BASE/logs/train_${DATASET}_${VARIANT}_${POLICY}_s${SEED}.log`.

### Stage 5 — evaluate

For a registered task, evaluate a completed ACT checkpoint with
the same native vector scheduler at one or more environments. The initial-state
file determines the episode count; every state is consumed once without retries
or resampling. These tasks load ACT in the simulation process and require
LeRobot in that Python environment. ACT must use one observation and no temporal
ensembling. Each environment keeps its own action queue.
Vector ACT evaluation executes 100 actions per inference by default. Set
`N_ACTION_STEPS` to override it; the value must not exceed the checkpoint chunk size.

```bash
INIT_STATES=meta/init_states/texture_sort.json NUM_ENVS=1 \
    OUT="$HOME/outputs/evaluation/texture_sort/vision/envs_1/run_01" \
    bash scripts/run_eval.sh texture_sort "$CKPT"

INIT_STATES=meta/init_states/texture_sort.json NUM_ENVS=4 \
    OUT="$HOME/outputs/evaluation/texture_sort/vision/envs_4/run_01" \
    bash scripts/run_eval.sh texture_sort "$CKPT"
```

Use the complete final `060000/pretrained_model` directory for texture sort and
`050000/pretrained_model` for stack cube. Repeat for vision, tacrgb and tacff.
Run on an available simulation host with Isaac and LeRobot installed.
Set `DEFAULT_PY` and `GPU` for that host.
Run a model's single and vector evaluations sequentially on the same GPU.

For both tasks, task position conditions must remain true
continuously for one second, with a prior valid grasp lasting at least 0.5 seconds.
Stack cube checks the relative horizontal position and height of the two cubes.
Texture sort checks that the nearest cup is the correct one and the object's
horizontal distance from its centre is less than 40 mm; height is not checked.
Gripper opening and release are not required. Success is latched, and only the
completed environment stops; other environments continue. Runs that reach the
time limit without success fail. Expert collection uses the same criterion and
may finish its remaining scripted phases after success.

The default output is `$HOME/outputs/evaluation/<task>/run_01`. Set `OUT` to
select a distinct directory for another checkpoint, variant or run.

Each output directory contains `eval_report.json`, `configuration.json`,
`checkpoint_sha256.json`, and per-episode starts, terminal states and results
under `attempts/`. Reports include success rates, confidence intervals, state
identities and timing. A failed or interrupted run has no final success rate.
Use a new output directory for each run.

For visual review, set `VIDEO=1` and a separate output directory. This writes
`visual_review/mosaic.mp4` and frame metadata with slot and initial-state IDs.
Use a fixed subset initial-state file for shorter review runs. Keep video disabled
in performance comparisons, and retain all 100 published states for success rates.

#### Evaluate with existing initial states

Every evaluation uses a saved initial-state file. Start with the published
initial-state set for the task (100 states for texture sort and stack cube):

```bash
CKPT="$OUTPUT_ROOT/outputs/merged_demonstrations_tacff_act_s1000/checkpoints/last/pretrained_model"
INIT_STATES=meta/init_states/stack_cube.json VIDEO=1 \
    OUT="$OUTPUT_ROOT/eval/merged_demonstrations_tacff_act_s1000" \
    bash scripts/run_eval.sh stack_cube "$CKPT"
```

For four environments on two GPUs, use the same state file:

```bash
INIT_STATES=meta/init_states/stack_cube.json \
    NUM_ENVS=4 GPUS=0,1 SEED=11 VIDEO=1 \
    OUT="$OUTPUT_ROOT/eval/merged_demonstrations_tacff_act_s1000_parallel" \
    bash scripts/run_eval.sh stack_cube "$CKPT"
```

Both commands evaluate every state once. `NUM_ENVS=4 GPUS=0` places all four
environments on one GPU if memory permits. A missing state file stops the run.

| Option | Meaning / default |
| --- | --- |
| `INIT_STATES` | Initial-state JSON file; required |
| `NUM_ENVS` | Concurrent environments; `1` |
| `GPUS` | GPU IDs, distributed round-robin; `0` |
| `SEED` | Policy evaluation seed; `11`; does not select the initial states |
| `OUT` | Evaluation output directory |
| `VIDEO=1` | Save rollout videos |

For the parallel example:

| Output | Path |
| --- | --- |
| Combined report | `$OUTPUT_ROOT/eval/merged_demonstrations_tacff_act_s1000_parallel/eval_report.json` |
| Videos by state ID | `$OUTPUT_ROOT/eval/merged_demonstrations_tacff_act_s1000_parallel/videos/` |
| Worker logs | `$OUTPUT_ROOT/eval/merged_demonstrations_tacff_act_s1000_parallel/workers/worker_<id>/` |

The report includes success rate, per-state success and failure details,
and the evaluation-set hash. Missing or duplicate states mark a run incomplete.
Compare results using the same evaluation set and benchmark configuration.

#### Generate new initial states

To create a separate evaluation set, sample up to 80 candidate starts and
keep 50 that the scripted expert completes:

```bash
python benchmark/run/make_init_states.py --task stack_cube \
    --count 50 --max-candidates 80 --seed 0 --expert --num_envs 4 \
    --headless --enable_cameras \
    --out "$OUTPUT_ROOT/custom_states/stack_cube.json"
```

Each candidate is settled, frozen and restored before the expert runs. Only
successful candidates enter the final set. To generate without an expert:

```bash
python benchmark/run/make_init_states.py --task stack_cube \
    --count 50 --max-candidates 80 --seed 1 --no-expert --num_envs 4 \
    --out "$OUTPUT_ROOT/custom_states/stack_cube_unfiltered.json" \
    --headless --enable_cameras
```

Then evaluate using the new file:

```bash
INIT_STATES="$OUTPUT_ROOT/custom_states/stack_cube.json" \
    NUM_ENVS=4 GPUS=0,1 VIDEO=1 \
    OUT="$OUTPUT_ROOT/eval/stack_cube_custom" \
    bash scripts/run_eval.sh stack_cube "$CKPT"
```

An incomplete run writes a `.partial.json` file and exits unsuccessfully.
Use `--exclude <training_states.json>` to exclude exact training starts.
Texture-sort paired collection and independent evaluation sampling are described
in [the task example](example/texture_sort/README.md).

The new set is saved separately from the published set. Its scores are
comparable only with evaluations using the same states and configuration.

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
                    envs/ (scenes) and tasks/ (experts, registry),
                    init_state.py (frozen evaluation states)
meta/               data the benchmark is pinned to: init_states/ (the
                    evaluation set) and baseline/ (the expert's own score)
src/tactile_codec/  quantisation profiles (float mm <-> uint16)
src/lerobot_tactile_ext/  ffv1 codec + LeRobot v3 dataset writer
src/gym_airseai/    the tasks as a gym environment, for lerobot
src/lerobot_airseai/  the matching lerobot environment config
scripts/run_*.sh    the pipeline entry points (collect, convert, train, eval)
                    plus install_airseai.sh
benchmark/run/      launch_env, collect_dataset, make_init_states, build_asset,
                    render_previews, make_* (task assets), verify_codec
benchmark/tasks/    run_expert.py, the shared expert runner
benchmark/teleop/   run_teleop.py, demonstrations driven by hand
benchmark/policy/   BC pipeline: dataset conversion, policy server
benchmark/eval/     evaluate.py, the policy evaluator
.githooks/          shared git hooks (DCO sign-off)
```

## 7. Simulation vs. the real robot

The hardware the simulation models:

| subsystem | what it is |
| --- | --- |
| arms | OpenArm dual-arm, 7 DoF each, leader-follower teleoperation for collection |
| end effector | two-finger electric gripper per arm |
| cameras | RealSense D435 at the head, a D405 at each wrist |
| tactile | vision-based fingertips (Daimon DM-Tac), four of them |
| compute | Jetson AGX Orin on the robot, RTX workstation for training |

It records the same cameras, state and action as the simulation, and these
tactile keys per sensor:

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
