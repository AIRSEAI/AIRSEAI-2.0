# Texture sort paired demonstrations

The normal task workflow is `make_init_states.py` followed by
`collect_dataset.py --init_states`. This example adds surface variants between
those steps and retains only complete successful pairs after collection.

```bash
RENDER=1 bash example/texture_sort/collect.sh \
    "$OUTPUT_ROOT/raw/texture_sort_pairs_s41_p250" \
    "$OUTPUT_ROOT/init_states/texture_sort/v1/train_pairs_s41_p250.json"
```

Defaults are 250 successful pairs / 500 episodes, four environments, seed 41,
32 base states per batch, and at most 2,000 paired candidate episodes. Override
`PAIRS`, `NUM_ENVS`, `SEED`, `BATCH_PAIRS`, `MAX_CANDIDATES`, `GPU`, `REPORT_DIR`,
`WRITE_MODE`, `WRITE_QUEUE_SIZE`, `RENDER`, or `DEFAULT_PY` as needed.
Each batch uses `SEED + batch_index`; its inputs and reports are retained.
Changing batch size changes the sampling sequence.

The example performs these steps:

1. Generate independent frozen base states through the generic entry without an expert.
2. Expand each state into smooth/rough variants with the same robot state and layout.
3. Collect trajectories through the generic multi-environment expert entry.
4. Keep both members only when both succeed and finish writing; draw another batch
   if necessary. A completed dataset is published only when the quota is reached.

Surface variants select the task's authored assets, swapping the active and
parked puck poses and updating `target_key`. They do not merely attach a material
label. Cup poses and robot positions/targets remain unchanged. Both members share
an explicit expert RNG stream. Their subsequent physical trajectories can differ.
Base state success is not inherited by either variant.

To use the individual entry points:

```bash
python benchmark/run/make_init_states.py --task texture_sort \
    --no-expert --count 8 --max-candidates 32 --seed 41 --num_envs 4 \
    --out "$OUTPUT_ROOT/base_states.json" --headless --enable_cameras

python example/texture_sort/generate_pairs.py \
    --src "$OUTPUT_ROOT/base_states.json" --seed 41 \
    --out "$OUTPUT_ROOT/pair_states.json"

python benchmark/run/collect_dataset.py --task texture_sort \
    --init_states "$OUTPUT_ROOT/pair_states.json" --num_envs 4 --gpus 0 \
    --dataset "$OUTPUT_ROOT/raw/pair_attempts" --render
```

The standalone collector retains individual successes. Use `collect.sh` for
complete-pair filtering and replenishment. Pair IDs, base IDs and variant labels
remain metadata, never policy input features. The final manifest maps dataset
episode indices to frozen state IDs. Individual candidate reports retain failures.
An exhausted budget writes `.partial.json`, returns exit code 2, and does not
publish the final dataset. Operational failures stop the workflow; existing batch
artifacts remain available. Choose new paths for another run.

With `RENDER=1`, each batch writes a fixed-camera mosaic to
`<report_dir>/batch_NNNN/collection/texture_sort.mp4`, with frame metadata beside it.

## Independent evaluation states

Generate 100 expert-success-filtered starts through the generic entry:

```bash
python benchmark/run/make_init_states.py --task texture_sort \
    --expert --count 100 --max-candidates 1000 --seed 42 --num_envs 4 \
    --exclude "$OUTPUT_ROOT/init_states/texture_sort/v1/train_pairs_s41_p250.json" \
    --out "$OUTPUT_ROOT/init_states/texture_sort/v1/eval_s42_n100_expert_checked.json" \
    --headless --enable_cameras
```

There is no pairing, class quota or example transformation for evaluation.
Class frequencies follow task sampling and expert acceptance. The saved states
are the frozen starts restored before the expert runs, not terminal states.
This is an expert-solvable subset. `stack_cube` uses the same generic entry.

## Conversion

```bash
DATASET=texture_sort_pairs_s41_p250 VARIANTS="vision tacrgb tacff" \
    bash scripts/run_convert.sh "$OUTPUT_ROOT/raw/texture_sort_pairs_s41_p250"
```

Set `BASE` to the converted dataset output directory and `DEFAULT_PY` to the
training environment's Python executable. Keep reference datasets read-only
and verify checksums when transferring raw recordings.

## Training

Train each converted variant with the same ACT recipe:

```bash
VARIANT=tacrgb GPU=6 \
    DS="<converted_dataset_root>/texture_sort_pairs_s41_p250_tacrgb" \
    BASE="<task_output_root>/stage4" \
    bash example/texture_sort/train.sh
```

Choose an allocated GPU and repeat for `vision` and `tacff` with the matching
dataset paths. Each invocation trains one variant on all 500 episodes, including
both members of all 250 pairs.

Defaults are 60,000 steps, batch size 8, seed 1000, eight loader workers,
checkpoints every 20,000 steps and logs every 200 steps. Override `STEPS`, `BATCH`,
`SEED`, `WORKERS`, `SAVE_FREQ`, `LOG_FREQ`, `DATASET`, `DS`, `BASE`, `GPU`, or
`DEFAULT_PY` through the environment. The wrapper calls the generic training
entry with explicit ACT settings: one observation, 100 predicted and executed
actions, ImageNet-pretrained ResNet18, learning rate 1e-5 for both backbone and
policy, weight decay 1e-4, KL weight 10, and AMP disabled.
Additional command-line arguments are passed to the LeRobot training entry.

With the default dataset name, outputs are:

```text
<BASE>/outputs/texture_sort_pairs_s41_p250_<variant>_act_s1000/checkpoints/
<BASE>/logs/train_texture_sort_pairs_s41_p250_<variant>_act_s1000.log
```

After training, transfer the complete `060000/pretrained_model` directory,
including weights, configuration and preprocessing files, to the evaluation
host. Keep optimizer state on the training host for resuming training. Verify
file checksums and policy loading before evaluation. An evaluation destination is
`<evaluation_root>/checkpoints/texture_sort/<variant>/act_s1000/060000/pretrained_model/`.
Stack-cube models use the same destination structure with task `stack_cube` and
step `050000`. Model deployment does not start evaluation automatically.
