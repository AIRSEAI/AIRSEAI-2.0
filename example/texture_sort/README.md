# Texture sort: demonstration data design

This example uses `texture_sort` to show how demonstration diversity and dataset
composition can affect task success. Varying layouts provides different starting
conditions; pairing smooth and rough objects within the same layout helps avoid
associating a placement decision with position alone. The effect should be measured
on a separate, fixed evaluation set.

## 1. Dataset composition

The default dataset contains **250 successful pairs, or 500 episodes**:

- Each pair starts from the same layout and robot state, with one smooth-object
  episode and one rough-object episode. The object variant and its sorting target
  change; cup positions remain the same.
- Different pairs use independently sampled base layouts.
- Both episodes must pass expert verification to retain a pair. Incomplete pairs
  are excluded, and new candidates are collected until the target is reached or
  the candidate budget is exhausted.

This produces 250 smooth and 250 rough episodes. Collection defaults to four
parallel environments, seed 41, and at most 2,000 candidate episodes. `PAIRS`
controls the number of successful pairs, unlike the attempt count in the generic
collection script. Pair labels are metadata, not policy inputs.

## 2. Collect, convert and train

Run these commands from the repository root with the `airseai` environment active
and `OUTPUT_ROOT` set. Select an available GPU for each command that uses one.

### Collect paired demonstrations

```bash
PAIRS=250 NUM_ENVS=4 SEED=41 GPU=0 \
    bash example/texture_sort/collect.sh \
    "$OUTPUT_ROOT/raw/texture_sort_pairs_s41_p250" \
    "$OUTPUT_ROOT/init_states/texture_sort/train_pairs_s41_p250.json"
```

`collect.sh` runs `collect_pairs.py`, which generates frozen base states through
`benchmark/run/make_init_states.py`, expands them with
`example/texture_sort/generate_pairs.py`, and collects expert trajectories through
`benchmark/run/collect_dataset.py`. It then selects complete successful pairs.
The second output path stores the training initial states.

### Convert to the three input modes

```bash
DATASET=texture_sort_pairs_s41_p250 VARIANTS="vision tacrgb tacff" \
    BASE="$OUTPUT_ROOT/datasets" \
    bash scripts/run_convert.sh "$OUTPUT_ROOT/raw/texture_sort_pairs_s41_p250"
```

The outputs are `$OUTPUT_ROOT/datasets/texture_sort_pairs_s41_p250_<variant>/`.
All three modes contain the same 500 episodes: `vision` uses cameras and robot
state, `tacrgb` adds tactile RGB, and `tacff` adds force-field images.

### Train ACT

```bash
for variant in vision tacrgb tacff; do
    DATASET=texture_sort_pairs_s41_p250 VARIANT="$variant" GPU=0 \
        DS="$OUTPUT_ROOT/datasets/texture_sort_pairs_s41_p250_${variant}" \
        BASE="$OUTPUT_ROOT" \
        bash example/texture_sort/train.sh || break
done
```

The example wrapper calls `scripts/run_train.sh` with the texture-sort ACT recipe:
60,000 steps, batch size 8, seed 1000, 100 actions per prediction and execution,
and learning rate 1e-5. Checkpoints are saved every 20,000 steps under
`$OUTPUT_ROOT/train/texture_sort_pairs_s41_p250_<variant>_act_s1000/checkpoints/`.

## 3. Evaluate through the common entry point

Use the same `scripts/run_eval.sh` workflow as other tasks. The published
`meta/init_states/texture_sort.json` contains 100 expert-verified evaluation starts.
These states are not paired. Each is evaluated once; use the same file for all
three models. If training and evaluation use different hosts, copy the complete
`pretrained_model` directory and update `CKPT` to its local path.

```bash
VARIANT=tacff
CKPT="$OUTPUT_ROOT/train/texture_sort_pairs_s41_p250_${VARIANT}_act_s1000/checkpoints/last/pretrained_model"

# Evaluate all fixed initial states in one environment
INIT_STATES=meta/init_states/texture_sort.json NUM_ENVS=1 GPU=0 \
    OUT="$OUTPUT_ROOT/eval/texture_sort_${VARIANT}_act_s1000_envs_1" \
    bash scripts/run_eval.sh texture_sort "$CKPT"

# Evaluate the same states across four environments
INIT_STATES=meta/init_states/texture_sort.json NUM_ENVS=4 GPU=0 \
    OUT="$OUTPUT_ROOT/eval/texture_sort_${VARIANT}_act_s1000_envs_4" \
    bash scripts/run_eval.sh texture_sort "$CKPT"
```

Repeat with `VARIANT=vision` and `VARIANT=tacrgb`, updating `CKPT` for each model.
Success rates and per-state results are saved in `$OUT/eval_report.json`. Add
`VIDEO=1` to save review videos; leave it disabled when comparing evaluation time.
