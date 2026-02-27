# StyleGAN2-ADA Fork: Required Modifications

> **Purpose:** This document describes source code changes to be made to
> the [Great-Bucket/stylegan2-ada-pytorch](https://github.com/Great-Bucket/stylegan2-ada-pytorch)
> fork before starting new training runs. Hand this file to the LLM working
> on that repo.

---

## Background

I am training StyleGAN2-ADA PyTorch on ~9,500 diverse 1024×1024 super-8
film frames. I have a separate operational project (GANesis) that manages
training runs on AWS EC2 (8× A100 GPUs). This repo is my fork of
dvschultz/stylegan2-ada-pytorch, which is itself a fork of
NVlabs/stylegan2-ada-pytorch.

I need to make specific source code modifications before starting new
training runs. The modifications come from a training plan developed
after analysing ~50 historical training runs. The full training plan
lives in the GANesis project at `docs/training_plan_202602.md` but is
not needed here — all required changes are self-contained in this
document.

### What the fork already includes

The dvschultz fork adds: Top-K training (`--topk`), fakes saved as .jpg,
multiple interpolation options, vertical mirroring (`--mirrory`),
`--initstrength`, `--nkimg`, closed form factorization, network blending,
Rosinality conversion, projector extensions, output size modification,
flesh digressions script, and CLIP-based tools. The NVlabs base training
code is otherwise unmodified.

---

## Modification 1: Add a hard ceiling on augment_p (CRITICAL)

### Problem

In `training/training_loop.py`, the ADA augmentation probability
adjustment applies a floor of 0 but **no ceiling**:

```python
augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
```

The ADA paper (Karras et al. 2020, "Training Generative Adversarial
Networks with Limited Data") states that augmentation probability must
stay below 0.8 for the non-leaking guarantee to hold. When p exceeds
1.0, all augmentations become deterministic — `training/augment.py`
applies each augmentation when `torch.rand() < constant * self.p`,
which is always true when `self.p > 1`.

In my historical runs, augment_p reached values of **15–47** because
the code has no ceiling. This caused severe augmentation leaking
(rotated images, colour shifts appearing in generated output).

### Required change

Add a ceiling of 0.85 immediately after the existing floor operation.
The value 0.85 provides a small margin above the paper's 0.8 safe
threshold.

**In `training/training_loop.py`**, find the line (approximately line 324):

```python
augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
```

Change it so that a ceiling is also applied. Ideally this ceiling value
should be wired to a new CLI argument (see Modification 4 below), but
at minimum a hardcoded cap is acceptable.

### Warning on cap hit

When augment_p reaches the ceiling, print a warning to the training log
so it is visible in `log.txt`. Something like:

```
Warning: augment_p capped at 0.85. Discriminator may be overfitting — consider increasing --gamma.
```

This only needs to print once (or at most once per N ticks) to avoid
spamming the log.

---

## Modification 2: Verify augment_p is in stats.jsonl

### Context

The `log.txt` status line already prints the augment value. In the
NVlabs code (training_loop.py, approximately line 343), the same
`training_stats.report0('Progress/augment', ...)` call both prints
to stdout and feeds the stats collector that writes `stats.jsonl`.

### Required action

Verify that `Progress/augment` appears in `stats.jsonl` entries in this
fork. It should already be there from the NVlabs base code, but confirm
that dvschultz's modifications have not inadvertently changed the stats
collection. If it is missing, ensure the `training_stats.report0` call
for augment is present and functioning.

---

## Modification 3: Verify pr50k3_full metric works

### Context

The `pr50k3_full` metric (precision and recall for GANs) is already
implemented in the NVlabs codebase under `metrics/`. I need to pass
`--metrics=fid50k_full,pr50k3_full` on the command line to enable it
during training. No code change should be required.

### Required action

Verify that `calc_metrics.py --metrics=pr50k3_full` runs correctly
in this fork. dvschultz may have modified metric handling. Run a quick
test:

```bash
python calc_metrics.py --metrics=pr50k3_full \
  --network=/path/to/any/network-snapshot.pkl \
  --data=/path/to/dataset.zip --mirror=1
```

If it fails, debug and fix. If it works, no changes needed.

---

## Modification 4: Make augment_p ceiling a CLI argument (RECOMMENDED)

### What

Add a new command-line option in `train.py`:

```
--augcap    Maximum augmentation probability for ADA [default: 0.85]
```

### How

1. **In `train.py`**, add the CLI option definition near the other
   augmentation options (`--aug`, `--p`, `--target`, `--augpipe`):

   ```python
   @click.option('--augcap', help='Maximum augmentation probability for ADA [default: 0.85]', type=float)
   ```

2. **In `setup_training_loop_kwargs()`**, add the `augcap` parameter,
   validate it (must be between 0 and 1), and pass it through to the
   training loop kwargs.

3. **In `training/training_loop.py`**, accept the new parameter in the
   `training_loop()` function signature (with default 0.85), and use it
   in the augment_p ceiling from Modification 1.

### Why a CLI argument

This makes the ceiling tuneable per run without editing source code.
Different datasets or gamma values might warrant different caps. The
default of 0.85 is a good general value based on the ADA paper's < 0.8
guidance with a small margin.

---

## Modification 5: Verify Top-K training still works (NO CODE CHANGES)

### Context

The dvschultz fork includes `--topk={float value}` for experimental
Top-K training (from Sinha & Zhao, 2020). This modifies generator
training to only backpropagate gradients from the images the
discriminator was most unsure about.

### Required action

Verify that `--topk` still works by checking that the relevant code
paths are intact (likely in `training/loss.py` or similar). Do not
change any Top-K code unless something is broken. Document what
`--topk` values are reasonable (the original paper suggests starting
around 0.99 and decaying to ~0.5).

---

## What NOT to change

- Do not modify the generator or discriminator network architecture
- Do not change the loss function (`StyleGAN2Loss`)
- Do not modify the ADA target heuristic (the
  `signs_real - ada_target` calculation)
- Do not change default learning rates, EMA settings, or batch size
  logic in any config preset
- Do not remove any of dvschultz's existing additions
- Do not change the stats.jsonl format (only verify it works)

---

## Testing

After making changes, run the following to verify nothing is broken:

### Dry run (no GPU needed)

```bash
python train.py --outdir=/tmp/test --data=/path/to/any/dataset.zip \
  --gpus=1 --cfg=auto --dry-run
```

This should print the training configuration and exit without error.

### Short training run (1 GPU, minimal)

```bash
python train.py --outdir=/tmp/test --data=/path/to/any/small-dataset.zip \
  --gpus=1 --cfg=auto --kimg=2 --snap=1 --metrics=fid50k_full,pr50k3_full
```

This runs for 2 kimg and verifies that training starts, metrics compute,
and the augment_p ceiling code path is reached.

### Force the augment_p cap (to verify the ceiling works)

Run with an extremely low gamma to make ADA push augment_p up quickly:

```bash
python train.py --outdir=/tmp/test --data=/path/to/any/small-dataset.zip \
  --gpus=1 --cfg=auto --gamma=0.01 --kimg=5 --snap=1 --metrics=none
```

Within a few kimg, augment_p should hit the 0.85 ceiling and the
warning message should appear in the console output / log.txt. If
augment_p exceeds 0.85, the ceiling is not working.

---

## Summary of changes

| # | File(s) | Change | Priority |
|---|---------|--------|----------|
| 1 | `training/training_loop.py` | Add augment_p ceiling (0.85 default) + warning | Critical |
| 2 | `training/training_loop.py` | Verify `Progress/augment` in stats.jsonl | Verify only |
| 3 | `metrics/` | Verify `pr50k3_full` works | Verify only |
| 4 | `train.py` + `training/training_loop.py` | Add `--augcap` CLI argument | Recommended |
| 5 | `training/loss.py` (or similar) | Verify `--topk` works | Verify only |
