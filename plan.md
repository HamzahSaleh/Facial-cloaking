# Irreversible Facial Cloaking: Embedding Collapse + Frequency Entanglement

## Context

The project needs a cloaking algorithm that adds imperceptible perturbations to face images such that:
1. The original identity **cannot be retrieved** via cosine-similarity search
2. The cloak **cannot be removed** by denoising, JPEG compression, or neural purification
3. The image **remains visually identical** to the original (SSIM tunable via epsilon)

Existing methods (Fawkes, LowKey) fail at irreversibility because they add a thin, separable additive layer `x + delta` that can be estimated and subtracted. Our approach makes the perturbation **structurally inseparable** from identity-critical content.

## The Novel Idea: Three-Pronged Irreversibility

| Component | What it does | Why it's irreversible |
|---|---|---|
| **Embedding Collapse** | Pushes all cloaked embeddings toward a single "null attractor" point in CLIP space | Many-to-one: 100 identities all map to the same point; no inverse exists |
| **Frequency Entanglement** | Forces perturbation energy into mid-frequency DCT bands (bands 3-12) that encode identity (eye spacing, jawline, nose geometry) | Filtering the perturbation = destroying the face; they occupy the same frequencies |
| **Non-linear Channel Mixing** | Applies `softplus(W @ x + delta)` instead of `x + delta` | No closed-form inverse; can't "subtract delta" when a non-linearity wraps the sum |

## Files to Create

### 1. `src/facial_cloaking/embed.py` — CLIP embedding utilities

- `load_clip_model(device)` — wraps the existing `open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")` pattern from `scripts/verify_env.py:50-53`
- `encode_image(model, preprocess, img, device)` — PIL image -> L2-normalized 512-dim tensor
- `compute_null_attractor(model, preprocess, clean_test_csv, device)` — averages per-identity embeddings from clean_test split, then averages those 100 identity centroids into a single null point `e_null`
- `compute_identity_embeddings(model, preprocess, csv_path, device)` — returns dict `{identity_name: mean_embedding}`

Reuses: `facial_cloaking.paths.SPLITS_DIR`, `facial_cloaking.data.load_image`

### 2. `src/facial_cloaking/dct_utils.py` — Differentiable DCT in PyTorch

- `dct_basis_matrix(n=8)` — 8x8 DCT-II basis matrix
- `blockwise_dct2d(x, block_size=8)` — batched forward DCT via `unfold` + matrix multiply (no loops)
- `blockwise_idct2d(coeffs, block_size=8)` — batched inverse DCT via `fold`
- `mid_frequency_mask(block_size=8, low=3, high=12)` — JPEG zig-zag order soft mask selecting identity-critical bands
- `rgb_to_ycbcr(x)` / `ycbcr_to_rgb(x)` — differentiable 3x3 matrix color conversion

No new dependencies — pure torch.

### 3. `src/facial_cloaking/cloak.py` — Core cloaking optimization

**Dataclass**: `CloakConfig` with fields:
- `epsilon` (default 8/255), `n_steps` (300), `lr` (0.01)
- `lambda_embed` (1.0), `lambda_freq` (0.3), `lambda_quality` (2.0)
- `freq_low` (3), `freq_high` (12)

**Loss function** (jointly minimized per image):
```
L = lambda_embed * L_embed + lambda_freq * L_freq + lambda_quality * L_quality

L_embed  = cos_sim(e_cloak, e_identity) - cos_sim(e_cloak, e_null)   # collapse toward null
L_freq   = -||M_freq * DCT(x_cloak - x)||_2                          # concentrate in mid-freq
L_quality = 1 - SSIM(x_cloak, x)                                     # preserve appearance
```

**Optimization loop** (per image, ~300 steps of Adam):
```python
delta_dct = zeros(...)        # learnable DCT-domain perturbation
W = eye(3)                    # learnable 3x3 channel mixing matrix

for step in range(n_steps):
    delta_pixel = IDCT(M_freq * delta_dct)         # frequency-constrained perturbation
    x_cloak = clamp(softplus(W @ x + delta_pixel)) # non-linear mixing
    x_cloak = project_linf(x_cloak, x, epsilon)    # enforce budget
    loss = L_embed + L_freq + L_quality
    loss.backward(); optimizer.step()
```

Key functions:
- `cloak_image(model, preprocess, img, e_null, config, device)` -> cloaked PIL image + metrics dict
- `cloak_batch(model, preprocess, image_paths, e_null, config, device, output_dir)` -> saves all cloaked images

**SSIM**: use `skimage.metrics.structural_similarity` (already in requirements.txt as scikit-image). For differentiable SSIM in the loss, implement the windowed Gaussian formula directly in torch (standard ~20 lines).

**Constraint on W**: clip `||W - I||_F < 0.1` each step to prevent visible color shift.

### 4. `scripts/compute_attractors.py` — Precompute embeddings

- Loads CLIP, encodes all clean_test images, computes null attractor + per-identity embeddings
- Saves to `outputs/attractors.pt`
- Run once before cloaking

### 5. `scripts/run_cloaking.py` — Main entry point

```bash
python scripts/run_cloaking.py --epsilon 8 --steps 300 --lr 0.01
python scripts/run_cloaking.py --epsilon 16 --steps 500   # aggressive
```

- Reads `splits/protected.csv`, loads `outputs/attractors.pt`
- Calls `cloak_batch`, saves cloaked images to `outputs/cloaked/`
- Writes `outputs/cloaking_report.json` with per-image metrics (loss components, SSIM, cosine similarity before/after)

### 6. `scripts/evaluate.py` — Three-surface evaluation

**Surface 1 — Identity Retrieval**: Cosine similarity of cloaked embeddings against all 100 identity embeddings. Report Rank-1 accuracy (target: ~0%) and mean cosine sim to true identity.

**Surface 2 — Identity Q&A Resistance**: Proxy metric — gap between cosine sim to true identity vs. max cosine sim to any other identity. If cloaked is closer to wrong identity or null, Q&A fails.

**Surface 3 — Editing Resistance**: Hook for external editing pipeline. Measure cosine sim between edited output and original uncloaked identity.

**Visual Quality**: SSIM, PSNR between cloaked and original.

### 7. `scripts/robustness_test.py` — Prove irreversibility

Apply purification attacks to cloaked images, then re-run retrieval:
- JPEG compression (quality 50, 75, 95)
- Gaussian blur (sigma 1.0, 2.0)
- Bilateral filtering
- Bit-depth reduction (4-bit, 6-bit)

If identity retrieval stays near 0% after purification, the cloak is irreversible.

## Implementation Order

```
Phase 1 (parallel):  embed.py + dct_utils.py
Phase 2:             cloak.py (depends on both above)
Phase 3:             compute_attractors.py -> run_cloaking.py
Phase 4 (parallel):  evaluate.py + robustness_test.py
```

## Verification

1. Run `python scripts/compute_attractors.py` — should produce `outputs/attractors.pt`
2. Run `python scripts/run_cloaking.py --epsilon 8 --steps 50` on a small subset — check cloaked images exist and look visually similar
3. Run `python scripts/evaluate.py` — Rank-1 retrieval accuracy should drop dramatically vs. uncloaked baseline
4. Run `python scripts/robustness_test.py` — retrieval should NOT recover after JPEG/blur/filtering
5. Visually inspect cloaked images — should be indistinguishable from originals to human eyes

## Industry-Aligned Defaults

Use these defaults for baseline runs and only deviate when an ablation requires
it.

- Transform: 8x8 block DCT with JPEG zig-zag ordering
- Frequency entanglement target: mid-frequency indices 3-12
- Color policy: YCbCr (ITU-R BT.601), with stronger weighting on Y
- Perturbation budget baseline: epsilon = 8/255
- Budget ablations: epsilon in {4/255, 16/255}
- Visual metrics: SSIM and PSNR (LPIPS recommended)
- Identity metrics: Rank-1 retrieval and mean cosine to true identity
- Robustness tests: JPEG (95/75/50), Gaussian blur (1.0/2.0), bilateral,
    bit-depth reduction (6-bit/4-bit)
- Reporting: fixed random seeds, exact preprocessing details, and mean+std
    across identities/images, with uncloaked vs cloaked deltas side-by-side
