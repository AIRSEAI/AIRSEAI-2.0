# PICO VR collection validation

This guide covers the simulation-only keyboard/OpenXR collection path. It does
not connect the physical OpenArm hardware. Configure the supported Isaac
Sim/Isaac Lab environment and XR streaming service before starting an immersive
session; the Python unit tests do not establish a headset connection.

## Changes retained in PR #18

- Palm tracking and bimanual controller input, with continuous trigger-controlled
  gripper targets and a squeeze clutch for each arm.
- Tracking validity checks, stationary-pose holding, measured-TCP target
  rebasing, and bounded joint targets.
- Optional state/action-only diagnostics, failed-attempt retention, and final
  scene rendering after recording completes.
- Paused palm grippers remain frozen. Rejected IK commands restore the held
  gripper target. XR diagnostics record the effective right-B button action.

Controller input must include position and orientation validity flags in the
XR pose map. Unflagged matrix fallbacks are intentionally rejected; an active
device alone is insufficient to establish valid tracking. If controller input
is absent, inspect `AIRSEAI_XR_DIAG=1` output for pose names and validity flags
before changing this policy.

Ordinary native environments retain Isaac Lab's default texture-loading wait.
The earlier deployment's unconditional `wait_for_textures=False` override is
not restored.

## Native collection deadlines

`--horizon_s` overrides the native episode deadline. Without it, the selected
task's deadline is used. Both the override and the effective task default must
be finite and strictly positive. The effective value is recorded as
`effective_horizon_s` in `configuration.json`.

The deadline is measured in advancing simulation time. Paused recording or
invalid tracking does not advance this clock. `--xr-keep-open-s`, by contrast,
uses wall-clock time after recording has finished.

## CPU regression checks

Run in the installed AIRSEAI environment, including PyTorch and Gymnasium:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  python -B -m pytest -p no:cacheprovider tests -q
```

The VR tests replace the XR SDK and simulator boundary with CPU fixtures. They
exercise input validity, button edges, continuous grippers, pause/release,
stationary targets, IK rejection, joint bounds, native device parameters,
deadlines, and completed-scene rendering. They do not verify GPU rendering,
actual controller tracking, network streaming, or physical contact behavior.

## Optional immersive state/action check

Use a new output directory on every run. Start with low gains and a stable
scene, then test controller tracking before recording a manipulation attempt.

```bash
AIRSEAI_XR_DIAG=1 AIRSEAI_XR_B_FINISH_ONLY=1 \
  python benchmark/teleop/run_teleop.py --task stack_cube \
    --teleop_device vr --episodes 1 --horizon_s 180 \
    --vr-joint-speed-deg-s 30 --vr-position-gain 1 \
    --xr-state-only --xr-keep-open-s 30 \
    --out outputs/pico_state_check_001
```

If the configured streaming service requires the XR viewport during offscreen
rendering, set `AIRSEAI_XR_FORCE_VIEWPORT=1` as well. Streaming server and client
setup is separate from this command.

With two active controllers, X starts/pauses, squeeze clutches arm motion, and
the trigger continuously controls the corresponding gripper. With the B option
above, right B finishes without marking operator success; left Y discards the
attempt. Without that option, B marks operator success. Automatic criterion
success and operator marking remain separate in the reports.

Accept the immersive check only after observing stable images, valid tracking,
no motion with stationary inputs, continuous gripper control, successful
pause/resume and saving, and a retained final scene. Inspect the saved reports
and trajectory rather than treating a visible scene as a saved-data result.

## Full-data collection

Omit `--xr-state-only` and provide a new dataset directory to record the full
camera/tactile streams:

```bash
AIRSEAI_XR_DIAG=1 AIRSEAI_XR_B_FINISH_ONLY=1 \
  python benchmark/teleop/run_teleop.py --task stack_cube \
    --teleop_device vr --episodes 10 --horizon_s 180 \
    --keep-unclean --dataset outputs/raw/pico_stack_cube_001
```

`--episodes` counts attempts, not successful demonstrations. `--keep-unclean`
retains failed attempts with their original labels; it does not turn them into
training successes. A requirement for 50 successful demonstrations needs a
separate qualification/counting step. Preserve raw labels and apply any curated
success/failure mapping explicitly when exporting training inputs.

State-only outputs do not contain the full RGB/tactile dataset and must not be
passed to the normal dataset converter as if they did.
