from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image


def _load_local_viz_module():
    """Load local visualize_frequency_bands.py to avoid package-name collisions."""
    scripts_dir = Path(__file__).resolve().parent
    module_path = scripts_dir / "visualize_frequency_bands.py"
    spec = importlib.util.spec_from_file_location("local_visualize_frequency_bands", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


viz = _load_local_viz_module()


def _resolve_bands(
    low_mid_split: int,
    mid_high_split: int,
    band_min: int,
    band_max: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Resolve two boundaries and min/max into contiguous low/mid/high ranges."""
    band_min = int(np.clip(band_min, 0, 63))
    band_max = int(np.clip(band_max, 0, 63))
    if band_max < band_min:
        band_min, band_max = band_max, band_min

    # Need at least 3 indices to create low, mid, and high ranges.
    if band_max - band_min < 2:
        band_max = min(63, band_min + 2)
        band_min = max(0, band_max - 2)

    low_mid_split = int(np.clip(low_mid_split, band_min, band_max - 2))
    mid_high_split = int(np.clip(mid_high_split, low_mid_split + 1, band_max - 1))

    low = (band_min, low_mid_split)
    mid = (low_mid_split + 1, mid_high_split)
    high = (mid_high_split + 1, band_max)
    return low, mid, high


def process_image(
    image: Image.Image,
    low_mid_split: int,
    mid_high_split: int,
    band_min: int,
    band_max: int,
    color_space: str,
    display_mode: str,
    display_percentile: float,
    low_gain: float,
    mid_gain: float,
    high_gain: float,
    y_weight: float,
    cbcr_weight: float,
    last_panel_state: Image.Image | None,
) -> tuple[Image.Image | None, Image.Image | None, str, str, Image.Image | None]:
    if image is None:
        return None, last_panel_state, "", "Upload an image to start.", last_panel_state

    image = image.convert("RGB")
    x_rgb = np.asarray(image).astype(np.float32) / 255.0

    if color_space == "ycbcr":
        x = viz.rgb_to_ycbcr_bt601(x_rgb)
        x = x.copy()
        x[..., 0] *= y_weight
        x[..., 1] *= cbcr_weight
        x[..., 2] *= cbcr_weight
    else:
        x = x_rgb

    low, mid, high = _resolve_bands(low_mid_split, mid_high_split, band_min, band_max)
    c = viz.dct_basis(8)

    low_img_arr = viz.reconstruct_band(x, c, low[0], low[1], block=8)
    mid_img_arr = viz.reconstruct_band(x, c, mid[0], mid[1], block=8)
    high_img_arr = viz.reconstruct_band(x, c, high[0], high[1], block=8)

    if display_mode == "signed":
        low_disp = viz.scale_signed_for_display(low_img_arr, gain=low_gain, percentile=display_percentile)
        mid_disp = viz.scale_signed_for_display(mid_img_arr, gain=mid_gain, percentile=display_percentile)
        high_disp = viz.scale_signed_for_display(high_img_arr, gain=high_gain, percentile=display_percentile)
    else:
        low_disp = viz.scale_magnitude_for_display(low_img_arr, gain=low_gain, percentile=display_percentile)
        mid_disp = viz.scale_magnitude_for_display(mid_img_arr, gain=mid_gain, percentile=display_percentile)
        high_disp = viz.scale_magnitude_for_display(high_img_arr, gain=high_gain, percentile=display_percentile)

    if color_space == "ycbcr":
        low_disp = viz.ycbcr_to_rgb_bt601(low_disp)
        mid_disp = viz.ycbcr_to_rgb_bt601(mid_disp)
        high_disp = viz.ycbcr_to_rgb_bt601(high_disp)

    original = viz.draw_label(image, "Original")
    low_img = viz.draw_label(viz.to_pil_uint8(low_disp), f"Low [{low[0]}-{low[1]}]")
    mid_img = viz.draw_label(viz.to_pil_uint8(mid_disp), f"Mid [{mid[0]}-{mid[1]}]")
    high_img = viz.draw_label(viz.to_pil_uint8(high_disp), f"High [{high[0]}-{high[1]}]")
    panel = viz.make_panel([original, low_img, mid_img, high_img], cols=2)

    stats = viz.band_energy_stats(x, c, block=8, low_range=low, mid_range=mid, high_range=high)
    payload = {
        "color_space": color_space,
        "display_mode": display_mode,
        "bands": {
            "low": [low[0], low[1]],
            "mid": [mid[0], mid[1]],
            "high": [high[0], high[1]],
        },
        "energy": stats,
    }

    summary = (
        f"Done. Energy share (%): low={stats['low_pct']:.2f}, "
        f"mid={stats['mid_pct']:.2f}, high={stats['high_pct']:.2f}, "
        f"residual={stats['residual_pct']:.2f}"
    )
    last_panel_for_display = last_panel_state
    new_last_panel_state = panel.copy()
    return panel, last_panel_for_display, json.dumps(payload, indent=2), summary, new_last_panel_state


def build_app() -> gr.Blocks:
    css = """
    #band_min_slider:hover::after,
    #band_max_slider:hover::after,
    #low_mid_slider:hover::after,
    #mid_high_slider:hover::after,
    #display_percentile_slider:hover::after,
    #low_gain_slider:hover::after,
    #mid_gain_slider:hover::after,
    #high_gain_slider:hover::after,
    #y_weight_slider:hover::after,
    #cbcr_weight_slider:hover::after {
      position: absolute;
      right: 0;
      top: -2.2em;
      background: #111;
      color: #fff;
      border: 1px solid #333;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
      z-index: 30;
      max-width: 320px;
      white-space: normal;
      pointer-events: none;
    }
    #band_min_slider:hover::after { content: "Band Min: lowest coefficient index included in the split."; }
    #band_max_slider:hover::after { content: "Band Max: highest coefficient index included in the split."; }
    #low_mid_slider:hover::after { content: "Low/Mid Split: low ends here; mid starts at next index."; }
    #mid_high_slider:hover::after { content: "Mid/High Split: mid ends here; high starts at next index."; }
    #display_percentile_slider:hover::after { content: "Display Percentile: higher values reveal weaker details but may add noise."; }
    #low_gain_slider:hover::after { content: "Low Gain: display amplification for the low-frequency panel."; }
    #mid_gain_slider:hover::after { content: "Mid Gain: display amplification for the mid-frequency panel."; }
    #high_gain_slider:hover::after { content: "High Gain: display amplification for the high-frequency panel."; }
    #y_weight_slider:hover::after { content: "Y Weight: luminance emphasis in YCbCr mode."; }
    #cbcr_weight_slider:hover::after { content: "Cb/Cr Weight: chroma emphasis in YCbCr mode."; }
    """

    with gr.Blocks(title="Frequency Band Visualizer", css=css) as demo:
        gr.Markdown("## Simple Frequency Band Visualizer")
        gr.Markdown("Upload an image, adjust the split sliders, then click Run.")
        gr.Markdown("Hover over a slider to see what it controls.")

        last_panel_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Input Image")

                band_min = gr.Slider(0, 63, value=0, step=1, label="Band Min", elem_id="band_min_slider")
                band_max = gr.Slider(0, 63, value=63, step=1, label="Band Max", elem_id="band_max_slider")
                low_mid_split = gr.Slider(0, 63, value=8, step=1, label="Low/Mid Split", elem_id="low_mid_slider")
                mid_high_split = gr.Slider(0, 63, value=20, step=1, label="Mid/High Split", elem_id="mid_high_slider")

                color_space = gr.Radio(["ycbcr", "rgb"], value="ycbcr", label="Color Space")
                display_mode = gr.Radio(["magnitude", "signed"], value="magnitude", label="Display Mode")

                display_percentile = gr.Slider(
                    90.0,
                    100.0,
                    value=99.5,
                    step=0.1,
                    label="Display Percentile",
                    elem_id="display_percentile_slider",
                )
                low_gain = gr.Slider(0.1, 20.0, value=1.0, step=0.1, label="Low Gain", elem_id="low_gain_slider")
                mid_gain = gr.Slider(0.1, 20.0, value=8.0, step=0.1, label="Mid Gain", elem_id="mid_gain_slider")
                high_gain = gr.Slider(0.1, 20.0, value=12.0, step=0.1, label="High Gain", elem_id="high_gain_slider")

                y_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Y Weight", elem_id="y_weight_slider")
                cbcr_weight = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="Cb/Cr Weight", elem_id="cbcr_weight_slider")

                run_btn = gr.Button("Run", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                panel = gr.Image(type="pil", label="Output Panel")
                last_panel = gr.Image(type="pil", label="Last Output Panel")
                stats = gr.Code(language="json", label="Stats")

        run_btn.click(
            fn=process_image,
            inputs=[
                image,
                low_mid_split,
                mid_high_split,
                band_min,
                band_max,
                color_space,
                display_mode,
                display_percentile,
                low_gain,
                mid_gain,
                high_gain,
                y_weight,
                cbcr_weight,
                last_panel_state,
            ],
            outputs=[panel, last_panel, stats, status, last_panel_state],
            show_progress="full",
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()
