"""Streamlit dashboard for the cloaking evaluation pipeline.

Run from the repo root:

    streamlit run scripts/eval_dashboard.py

The dashboard is a thin wrapper over ``facial_cloaking.eval_pipeline``:
it never re-implements scoring logic, only orchestrates UI state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

# Allow `streamlit run scripts/eval_dashboard.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from facial_cloaking import eval_metrics
from facial_cloaking.data import load_image
from facial_cloaking.embed import encode_pil_images, load_clip_model
from facial_cloaking.eval_pipeline import (
    Gallery,
    MethodSpec,
    resolve_method_rows,
    score_method,
)
from facial_cloaking.paths import PROJECT_ROOT
from facial_cloaking.purify import (
    bilateral_filter,
    bit_depth_reduction,
    gaussian_blur,
    jpeg_compress,
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP + gallery ...")
def _load_model_and_gallery(attractors_path: str, device: str):
    model, preprocess = load_clip_model(device)
    gallery = Gallery.load(Path(attractors_path), device=device)
    return model, preprocess, gallery


@st.cache_data(show_spinner=False)
def _discover_method_dirs(root: str) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(str(d) for d in p.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _build_purifications(
    *,
    use_jpeg: bool, jpeg_q: int,
    use_blur: bool, blur_sigma: float,
    use_bilateral: bool,
    use_bits: bool, bits: int,
) -> dict:
    purifs: dict = {}
    if use_jpeg:
        purifs[f"jpeg_q{jpeg_q}"] = jpeg_compress(jpeg_q)
    if use_blur:
        purifs[f"blur_s{blur_sigma}"] = gaussian_blur(blur_sigma)
    if use_bilateral:
        purifs["bilateral"] = bilateral_filter()
    if use_bits:
        purifs[f"bits{bits}"] = bit_depth_reduction(bits)
    return purifs


def _scores_to_dataframe(scores) -> pd.DataFrame:
    rows = []
    for s in scores:
        row = {
            "method": s.method,
            "n": s.n_images,
            "rank1": round(s.rank1, 3),
            "mean_true_cos": round(s.mean_true_cos, 3),
            "qa_gap": round(s.qa_gap, 3),
            "null_gap": round(s.null_gap, 3),
            "ssim": "—" if s.ssim != s.ssim else round(s.ssim, 3),  # NaN check
            "psnr": "—" if s.psnr != s.psnr else round(s.psnr, 2),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _purif_to_dataframe(scores) -> pd.DataFrame:
    purif_names = sorted({p for s in scores for p in s.purifications})
    if not purif_names:
        return pd.DataFrame()
    rows = []
    for s in scores:
        row = {"method": s.method}
        for pn in purif_names:
            m = s.purifications.get(pn)
            row[pn] = (
                f"{m['rank1']:.2f} / {m['mean_true_cos']:.2f}"
                if m is not None else "—"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _per_image_metrics(
    method_dir: Path | None,
    csv_path: Path,
    model, preprocess, gallery: Gallery, device: str,
) -> pd.DataFrame:
    """Encode each candidate individually and return per-image cosine sims."""
    spec = MethodSpec.uncloaked() if method_dir is None else MethodSpec.folder(
        method_dir.name, method_dir
    )
    rows = resolve_method_rows(spec, csv_path)
    if not rows:
        return pd.DataFrame()

    images = [load_image(r.candidate_path) for r in rows]
    embs = encode_pil_images(model, preprocess, images, device)
    sims_g = (embs @ gallery.embeddings.T).cpu()
    sims_n = (embs @ gallery.e_null.unsqueeze(0).T).cpu().squeeze(1)
    label_to_idx = {lab: i for i, lab in enumerate(gallery.labels)}

    out = []
    for i, r in enumerate(rows):
        true_i = label_to_idx.get(r.identity)
        true_sim = sims_g[i, true_i].item() if true_i is not None else float("nan")
        masked = sims_g[i].clone()
        if true_i is not None:
            masked[true_i] = float("-inf")
        max_other = masked.max().item()
        out.append({
            "identity": r.identity,
            "filename": r.filename,
            "original_path": str(r.original_path),
            "candidate_path": str(r.candidate_path),
            "cos_true": round(true_sim, 3),
            "cos_null": round(sims_n[i].item(), 3),
            "max_cos_other": round(max_other, 3),
            "qa_gap": round(true_sim - max_other, 3),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cloaking eval", layout="wide")
st.title("Facial-cloaking evaluation dashboard")

# ---- Sidebar: global config -------------------------------------------------
st.sidebar.header("Global")
csv_path = st.sidebar.text_input(
    "Eval CSV", value=str(PROJECT_ROOT / "splits" / "protected.csv")
)
attractors_path = st.sidebar.text_input(
    "Attractors", value=str(PROJECT_ROOT / "outputs" / "attractors.pt")
)
methods_root = st.sidebar.text_input(
    "Methods root", value=str(PROJECT_ROOT / "outputs" / "methods")
)
device = st.sidebar.selectbox(
    "Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1
)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=128, value=32)

if st.sidebar.button("Refresh method list"):
    _discover_method_dirs.clear()

if not Path(attractors_path).exists():
    st.warning(
        f"Attractors not found at {attractors_path}. "
        "Run `python scripts/compute_attractors.py` first."
    )
    st.stop()

model, preprocess, gallery = _load_model_and_gallery(attractors_path, device)
st.sidebar.success(f"Gallery: {len(gallery.labels)} identities, dim={gallery.embeddings.shape[1]}")

# ---- Method picker ----------------------------------------------------------
st.sidebar.header("Methods")
include_uncloaked = st.sidebar.checkbox("uncloaked (baseline)", value=True)
discovered = _discover_method_dirs(methods_root)
chosen_dirs = st.sidebar.multiselect(
    "Method folders",
    options=discovered,
    default=discovered,
    format_func=lambda p: f"{Path(p).name}  ({p})",
)
custom = st.sidebar.text_input("Custom method (name=path)", value="")

methods: list[MethodSpec] = []
if include_uncloaked:
    methods.append(MethodSpec.uncloaked())
for d in chosen_dirs:
    methods.append(MethodSpec.folder(Path(d).name, Path(d)))
if custom.strip():
    if "=" not in custom:
        st.sidebar.error("custom method must be 'name=path'")
    else:
        cn, cp = custom.split("=", 1)
        methods.append(MethodSpec.folder(cn.strip(), Path(cp.strip())))

# ---- Purification picker ----------------------------------------------------
st.sidebar.header("Purifications")
use_jpeg = st.sidebar.checkbox("JPEG compress", value=True)
jpeg_q = st.sidebar.slider("JPEG quality", 1, 100, 75, disabled=not use_jpeg)
use_blur = st.sidebar.checkbox("Gaussian blur", value=True)
blur_sigma = st.sidebar.slider("blur sigma", 0.0, 5.0, 1.0, 0.1, disabled=not use_blur)
use_bilateral = st.sidebar.checkbox("Bilateral filter", value=False)
use_bits = st.sidebar.checkbox("Bit-depth reduction", value=False)
bits = st.sidebar.slider("bits", 1, 8, 4, disabled=not use_bits)
purifs = _build_purifications(
    use_jpeg=use_jpeg, jpeg_q=jpeg_q,
    use_blur=use_blur, blur_sigma=blur_sigma,
    use_bilateral=use_bilateral,
    use_bits=use_bits, bits=int(bits),
)

# ---- Tabs -------------------------------------------------------------------
tab_score, tab_drill, tab_compare = st.tabs(["Score", "Drill-down", "Compare"])

# ----- Tab 1: Score ---------------------------------------------------------
with tab_score:
    st.subheader("Score selected methods")
    st.caption(
        f"{len(methods)} method(s), {len(purifs)} purification(s). "
        "Click Run to encode and score."
    )
    cols = st.columns([1, 1, 4])
    run = cols[0].button("Run", type="primary", disabled=not methods)
    save = cols[1].button("Save JSON", disabled="last_scores" not in st.session_state)

    if run:
        with st.spinner("Scoring ..."):
            scores = []
            prog = st.progress(0.0)
            for i, m in enumerate(methods):
                try:
                    s = score_method(
                        m,
                        csv_path=Path(csv_path),
                        gallery=gallery,
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        purifications=purifs or None,
                        batch_size=int(batch_size),
                    )
                    scores.append(s)
                except FileNotFoundError as e:
                    st.warning(f"{m.name}: {e}")
                prog.progress((i + 1) / len(methods))
            prog.empty()
        st.session_state.last_scores = scores

    if "last_scores" in st.session_state:
        scores = st.session_state.last_scores
        st.markdown("#### Identity surfaces + visual quality")
        st.dataframe(_scores_to_dataframe(scores), hide_index=True, use_container_width=True)
        purif_df = _purif_to_dataframe(scores)
        if not purif_df.empty:
            st.markdown("#### Robustness — `rank1 / mean_true_cos` under purification")
            st.dataframe(purif_df, hide_index=True, use_container_width=True)

    if save and "last_scores" in st.session_state:
        out = PROJECT_ROOT / "outputs" / "eval_dashboard_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {
                "csv": csv_path,
                "attractors": attractors_path,
                "device": device,
                "purifications": list(purifs.keys()),
                "scores": [s.to_dict() for s in st.session_state.last_scores],
            },
            indent=2,
        ))
        st.success(f"Saved {out}")

# ----- Tab 2: Drill-down ----------------------------------------------------
with tab_drill:
    st.subheader("Per-identity inspection")
    if not methods:
        st.info("Pick at least one method in the sidebar.")
    else:
        method_names = [m.name for m in methods]
        sel_name = st.selectbox("Method", method_names, key="drill_method")
        sel_method = next(m for m in methods if m.name == sel_name)

        per_img = _per_image_metrics(
            sel_method.candidate_dir, Path(csv_path),
            model, preprocess, gallery, device,
        )
        if per_img.empty:
            st.info("No candidate images found.")
        else:
            identities = sorted(per_img["identity"].unique())
            sel_id = st.selectbox("Identity", identities, key="drill_id")
            subset = per_img[per_img["identity"] == sel_id].reset_index(drop=True)
            st.dataframe(
                subset[["filename", "cos_true", "cos_null", "max_cos_other", "qa_gap"]],
                hide_index=True, use_container_width=True,
            )
            row = subset.iloc[0]
            cols = st.columns(2 + (1 if purifs else 0))
            with Image.open(row["original_path"]) as orig:
                cols[0].image(orig, caption=f"original\n{row['filename']}", use_container_width=True)
            with Image.open(row["candidate_path"]) as cand:
                cand = cand.convert("RGB").copy()
                cols[1].image(
                    cand,
                    caption=f"{sel_method.name}\ncos_true={row['cos_true']}  cos_null={row['cos_null']}",
                    use_container_width=True,
                )
                if purifs:
                    pname, pfn = next(iter(purifs.items()))
                    purified = pfn(cand)
                    p_emb = encode_pil_images(model, preprocess, [purified], device)
                    label_to_idx = {lab: i for i, lab in enumerate(gallery.labels)}
                    true_i = label_to_idx.get(sel_id)
                    cos_true_p = float((p_emb @ gallery.embeddings.T)[0, true_i]) if true_i is not None else float("nan")
                    cols[2].image(
                        purified,
                        caption=f"{pname}\ncos_true={cos_true_p:.3f}",
                        use_container_width=True,
                    )

# ----- Tab 3: Compare -------------------------------------------------------
with tab_compare:
    st.subheader("Side-by-side comparison")
    if "last_scores" not in st.session_state or len(st.session_state.last_scores) < 2:
        st.info("Run scoring on at least two methods first (Score tab).")
    else:
        scores = st.session_state.last_scores
        names = [s.method for s in scores]
        c1, c2 = st.columns(2)
        a = c1.selectbox("Method A", names, index=0, key="cmp_a")
        b = c2.selectbox("Method B", names, index=min(1, len(names) - 1), key="cmp_b")
        sa = next(s for s in scores if s.method == a)
        sb = next(s for s in scores if s.method == b)

        rows = []
        for field in ("rank1", "mean_true_cos", "qa_gap", "null_gap", "ssim", "psnr"):
            va = getattr(sa, field)
            vb = getattr(sb, field)
            delta = (vb - va) if (va == va and vb == vb) else float("nan")  # NaN-safe
            rows.append({
                "metric": field,
                a: round(va, 3) if va == va else "—",
                b: round(vb, 3) if vb == vb else "—",
                "Δ (B−A)": round(delta, 3) if delta == delta else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Robustness diff
        purif_names = sorted(set(sa.purifications) | set(sb.purifications))
        if purif_names:
            st.markdown("#### Robustness diff — `rank1` (B − A)")
            rrows = []
            for pn in purif_names:
                pa = sa.purifications.get(pn, {}).get("rank1")
                pb = sb.purifications.get(pn, {}).get("rank1")
                d = (pb - pa) if (pa is not None and pb is not None) else None
                rrows.append({
                    "purification": pn,
                    a: f"{pa:.3f}" if pa is not None else "—",
                    b: f"{pb:.3f}" if pb is not None else "—",
                    "Δ": f"{d:+.3f}" if d is not None else "—",
                })
            st.dataframe(pd.DataFrame(rrows), hide_index=True, use_container_width=True)
