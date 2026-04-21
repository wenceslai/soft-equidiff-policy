# Project: Soft Equivariant Diffusion Policy on Push-T

Student research project implementing and testing soft equivariant diffusion policies
(based on Wang et al. 2024 "Soft Equivariant Policy") on the Push-T benchmark.

---

## Goal

Compare equivariant vs soft-equivariant vs non-equivariant diffusion policies on Push-T.
Baseline (non-equivariant) diffusion achieves ~0.42 success rate.
LeRobot's official diffusion policy achieves ~0.64 (uses pretrained ResNet-18 encoder).

---

## Results So Far

| Method | Steps | Success Rate | Mean Coverage | Notes |
|---|---|---|---|---|
| base_diffusion | 150k | 0.42 | 0.730 | non-equivariant baseline |
| soft_constant | 200k | 0.22 | 0.501 | large UNet, buggy normalization |
| soft_constant | 250k | 0.12 | 0.467 | worse than 200k — overfitting signal |
| baseline_no_softening | 200k | 0.06–0.08 | ~0.386 | exact equivariance, large UNet |
| baseline_no_softening_bigger | 200k | 0.06 | 0.276 | exact equivariance, larger arch |
| equi_exact_small_unet | 125k | 0.02 | 0.244 | fixed norm, small UNet, no aug |
| equi_exact_small_unet_small_enc | 150k | 0.02 | 0.279 | fixed norm, small UNet+enc, no aug |

---

## Key Findings / Hypotheses

### 1. Push-T has no equivariant variation in training data (most important)
The goal T is always at the same fixed position and orientation in every episode.
Only the T block being pushed starts in varied positions. This means:
- The task is NOT equivariant in the training distribution (no rotated goals ever appear)
- Forcing C8 equivariance constrains the model to relate 8 orientations, 7 of which
  have zero training coverage → actively harmful, explains near-0% success
- Soft equivariance (0.22) was better than exact (0.06-0.08) precisely because the
  free paths could partially override the constraint and fit the actual distribution
- Fix: C4 rotation augmentation (rotate image + state + action together at training time)
  artificially creates equivariant variation and makes the task genuinely C4-equivariant

### 2. Per-component normalization broke equivariance (fixed)
The Normalizer applied independent min-max scaling to x and y components of state/action.
The escnn encoders treat inputs as irrep(1) — proper 2D vectors under rotation.
If x_range ≠ y_range after normalization, rotating a normalized vector no longer
corresponds to rotating the original vector → equivariance broken before data hits the model.
Fix (train.py build_dataset): use max(x_range, y_range) as a single shared scale for both
components, centered at each component's midpoint.
Note: workspace center turned out to be (254.8, 272.0) — NOT exactly (256,256), confirming
x and y stats were indeed different and the fix was necessary.

### 3. UNet was massively oversized vs baseline (partially fixed)
Equivariant config used unet_down_dims=(256,512,1024) while the baseline used (64,128,256).
The baseline comment claimed they were "identical" — they were not.
The equivariant UNet was ~16x larger in parameters, harder to train, slower to converge.
Added --unet_down_dims CLI arg to train.py so this can be controlled per-run.
Current experiments use (64,128,256) to match the baseline.

### 4. Exact equivariance is too constrained for Push-T without augmentation
Schur's lemma means equivariant linear layers have far fewer free parameters than
unconstrained layers (e.g. action encoder: ~32 free scalars vs 64 unconstrained weights).
Without augmentation, this is a capacity restriction with no upside.
With augmentation making the task genuinely equivariant, this becomes the intended
inductive bias: model shares parameters across all 4/8 orientations → more sample efficient.

### 5. The "no augmentation needed" claim only holds when data is already equivariant
Equivariant-by-construction means: if trained on orientation X, correctly handles Y.
It does NOT mean: produces the correct output for orientation Y.
If the task ground truth doesn't respect the symmetry (fixed goal), the equivariant
model gives consistent but WRONG answers for unseen orientations.
This is a meaningful finding for the thesis.

---

## Code Changes Made

### train.py
- `build_dataset`: split into train/val by episode index (last 10% = val).
  Returns (train_dataset, val_dataset, stats). Stats from full dataset.
- `build_dataset`: equivariant normalization fix — uses shared scale for x and y.
- Added `--unet_down_dims` CLI arg (default still 256 512 1024, pass 64 128 256 for smaller).
- Added `--val_every` (default 10000) and `--val_batches` (default 32) CLI args.
- Added `_compute_val_loss`: switches to eval(), runs n_batches with torch.no_grad(), logs val/mse_loss.
- Added `--rot_aug` flag: enables C4 rotation augmentation on training batches (not val).
- Import C4Augmentation from augmentation.py; instantiated once, workspace center from stats.

### baseline_diffusion.py
- Same train/val split logic added to `_build_dataset` (10% val holdout).
- Added `_compute_val_loss` helper.
- Added `--val_every` and `--val_batches` CLI args to train subparser.
- val/mse_loss logged to wandb every val_every steps — comparable to train.py runs.

### augmentation.py (new file)
- C4Augmentation class: applies random C4 rotation to each batch sample.
- Rotates images with torch.rot90 (exact, free — no interpolation needed for 90° steps).
- Rotates 2D state/action vectors around workspace_center using rotation matrix.
- Groups samples by rotation index (4 groups) to avoid per-sample Python loops.
- Applied in training loop after .to(device), before policy(batch). Val batches skip it.

---

## Active Experiments

- equi_exact_c4_rotaug: exact C4 equivariance + C4 aug, N=4, small UNet (64,128,256)
- soft_constant_c4_rotaug: soft equivariance λ=0.1 + C4 aug, N=4, small UNet

---

## Architecture Notes

- Image encoder: 4-stage equivariant ResNet (from EquiDiff), adapted for 96×96 Push-T images.
  Designed for real-world robot images — likely heavier than needed for Push-T's simple synthetics.
  Controlled via --n_hidden (default 64 fields per group element).
- UNet: 1D temporal ConditionalUnet1D, shared across N group elements (processed via batch concat).
  obs_cond_dim = n_obs_steps * (n_hidden + state_features) = 2*(64+32) = 192 (same as baseline).
- Soft equivariance: SoftEquiWrapper adds a free (unconstrained) path alongside each escnn layer.
  Init std=0.01 so training starts near-equivariant. Penalty: λ * ||W_free||².
- Penalty modes: "constant" (λ fixed) or "step_dependent" (λ scales with diffusion timestep k).

## Useful Commands

```bash
# Baseline (non-equivariant)
python -m soft_equidiff.baseline_diffusion train --run_name base_diffusion --num_steps 250000

# Exact equivariance + C4 aug (current best hypothesis)
python -m soft_equidiff.train \
  --run_name equi_exact_c4_rotaug \
  --no_soften_image --no_soften_state --no_soften_action --no_soften_decoder \
  --N 4 --n_hidden 64 --unet_down_dims 64 128 256 \
  --rot_aug --num_steps 250000

# Soft equivariance + C4 aug
python -m soft_equidiff.train \
  --run_name soft_constant_c4_rotaug \
  --penalty_mode constant --lambda_base 0.1 \
  --N 4 --n_hidden 64 --unet_down_dims 64 128 256 \
  --rot_aug --num_steps 250000

# Eval single checkpoint
python -m soft_equidiff.eval_success_rate \
  --checkpoint outputs/RUN_NAME/policy_step0250000.pt \
  --n_episodes 50 --device cuda

# Eval all checkpoints in a run
python -m soft_equidiff.eval_success_rate \
  --checkpoint $(ls outputs/RUN_NAME/policy_step*.pt | sort | tr '\n' ',' | sed 's/,$//') \
  --n_episodes 50 --device cuda
```


# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.