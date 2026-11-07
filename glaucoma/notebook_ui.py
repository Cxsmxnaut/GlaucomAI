"""An in-notebook judging dashboard. No tunnel, web server or disk uploads."""

import base64
from html import escape
from io import BytesIO

from IPython.display import HTML, display
import ipywidgets as widgets
import torch

from glaucoma.pipeline import InputError, ModelError, decode_image, infer, load_model

STYLE = """
<style>
.gai {font-family:Inter,Arial,sans-serif;color:#193e43;line-height:1.55;max-width:1100px;background:#fff;padding:18px;border-radius:16px}
.gai * {box-sizing:border-box}
.gai .hero {background:#113f40;color:#fff;padding:30px 34px;border-radius:18px;margin:0 0 20px}
.gai .eyebrow {font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#a4ded0}
.gai h1 {font-size:32px;letter-spacing:-1px;margin:8px 0;color:#fff;font-weight:650}
.gai h2 {font-size:21px;margin:7px 0 12px;color:#193e43}
.gai p {margin:7px 0}
.gai .sub {color:#c9dfdc;font-size:14px}
.gai .grid {display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:16px 0}
.gai .card {background:#f6f9fa;border:1px solid #dce8e6;border-radius:14px;padding:24px}
.gai .image {background:#101e26;border-radius:10px;min-height:260px;display:flex;align-items:center;justify-content:center;overflow:hidden}
.gai img {display:block;max-height:340px;max-width:100%;object-fit:contain}
.gai .label {font-size:11px;letter-spacing:1.4px;text-transform:uppercase;color:#607c80}
.gai .score {font-size:48px;font-weight:650;letter-spacing:-2px;color:#147d70}
.gai .track {height:14px;border-radius:20px;background:#e0ebe9;overflow:hidden;margin:10px 0}
.gai .fill {height:100%;border-radius:20px;background:#147d70}
.gai .row {display:flex;justify-content:space-between;gap:16px;font-size:13px;margin:8px 0}
.gai .note {border-left:3px solid #c99341;padding:12px 16px;background:#fff8ec;color:#6e542d;font-size:12px;border-radius:4px;margin:16px 0}
.gai .muted {font-size:12px;color:#607c80}
.gai .pill {display:inline-block;padding:4px 10px;background:#e2efeb;border-radius:20px;color:#266f62;font-size:11px}
@media(max-width:700px){.gai .grid{grid-template-columns:1fr}.gai .hero{padding:22px}.gai h1{font-size:27px}}
</style>
"""

DISCLAIMER = """<div class="note"><b>Research prototype · not a diagnosis.</b>
This model has not been clinically validated. Scores can be wrong and are not calibrated probabilities
of disease. Non-referable does not mean healthy. A qualified eye-care professional must interpret
retinal findings. The app does not check whether an upload is a fundus photograph.</div>"""


def uploaded_file(value):
    """Normalize both Colab's ipywidgets 7 mapping and ipywidgets 8 sequence."""
    if not value:
        return None
    if isinstance(value, dict):
        name, file = next(iter(value.items()))
        return name, bytes(file["content"])
    file = value[0]
    return file["name"], bytes(file["content"])


def image_data_url(image):
    preview = image.copy()
    preview.thumbnail((800, 800))
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def result_html(image, result):
    score = result["referable"] * 100
    decision = "Above the referral-score threshold" if result["flagged"] else "Below the referral-score threshold"
    return STYLE + f"""<div class="gai"><div class="grid">
<div class="card"><div class="label">01 / Uploaded fundus image</div>
<div class="image"><img src="{image_data_url(image)}" alt="Uploaded image for research analysis"/></div>
<p class="muted">{image.width} × {image.height} original · 224 × 224 model input</p></div>
<div class="card"><div class="label">02 / Model output</div><h2>{decision}</h2>
<div class="score">{score:.1f}<span style="font-size:24px">%</span></div>
<p>Referable glaucoma model score</p>
<div class="track" role="progressbar" aria-label="Referable glaucoma score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score:.1f}"><div class="fill" style="width:{score:.2f}%"></div></div>
<div class="row"><span>Non-referable score</span><b>{result['non_referable']:.1%}</b></div>
<div class="row"><span>Referral-score threshold</span><b>{result['threshold']:.1%}</b></div>
<div class="row"><span>CPU preprocessing + inference</span><b>{result['elapsed_ms']:.0f} ms</b></div>
<p class="muted">The threshold was selected on validation images. This is a research routing signal, not a treatment recommendation.</p>
</div></div>{DISCLAIMER}</div>"""


def display_assistant(model_path="models/best_model.pth"):
    """Call once in Colab/Jupyter after training; session memory only."""
    torch.set_num_threads(2)
    loaded = load_model(model_path)
    display(HTML(STYLE + f"""<div class="gai"><div class="hero">
<div class="eyebrow">GlaucomAI / AI + Health</div>
<h1>A clearer view. An earlier signal.</h1>
<p class="sub">Explore glaucoma screening with a retinal photograph and a trained EfficientNet-B0.</p>
<span class="pill">Trained checkpoint connected · CPU inference</span></div>
<p><b>Upload a fundus photograph</b> <span class="muted">PNG or JPEG · under 10 MB · one image at a time</span></p>
{DISCLAIMER}</div>"""))
    uploader = widgets.FileUpload(accept=".png,.jpg,.jpeg", multiple=False, description="Choose image")
    analyze = widgets.Button(description="Analyze image →", button_style="success", disabled=True,
                             layout=widgets.Layout(width="180px"))
    reset = widgets.Button(description="Clear image", icon="trash")
    output = widgets.Output()
    state = {"image": None}

    def clear_image():
        if state["image"] is not None:
            state["image"].close()
        state["image"] = None
        analyze.disabled = True
        output.clear_output()

    def uploaded(change):
        clear_image()
        if not uploader.value:
            return
        try:
            name, content = uploaded_file(uploader.value)
            state["image"] = decode_image(content, name)
            analyze.disabled = False
            with output:
                display(HTML(STYLE + f'<div class="gai"><p class="pill">Image decoded · ready to analyze</p><div class="image"><img src="{image_data_url(state["image"])}" alt="Uploaded image preview"/></div></div>'))
        except InputError as exc:
            with output:
                display(HTML(f'<div class="gai"><div class="note">{escape(str(exc))}</div></div>'))

    def run_analysis(button):
        if state["image"] is None:
            return
        analyze.disabled = True
        analyze.description = "Analyzing…"
        output.clear_output(wait=True)
        try:
            result = infer(state["image"], loaded)
            with output:
                display(HTML(result_html(state["image"], result)))
        except (ModelError, RuntimeError) as exc:
            with output:
                display(HTML(f'<div class="gai"><div class="note">Analysis unavailable: {escape(str(exc))}</div></div>'))
        finally:
            analyze.description = "Analyze image →"
            analyze.disabled = state["image"] is None

    def reset_upload(button):
        clear_image()
        if isinstance(uploader.value, dict):
            # ipywidgets 7 computes its read-only value from these synced traits.
            uploader.metadata = []
            uploader.data = []
            uploader._counter = 0
            uploader.set_trait("value", {})
        else:
            uploader.value = ()

    uploader.observe(uploaded, names="value")
    analyze.on_click(run_analysis)
    reset.on_click(reset_upload)
    display(widgets.VBox([widgets.HBox([uploader, analyze, reset]), output]))
    return {"uploader": uploader, "analyze": analyze, "reset": reset, "output": output}
