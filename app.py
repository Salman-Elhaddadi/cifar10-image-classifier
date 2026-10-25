from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms as T

from model import CiFAR10

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱", "deer": "🦌",
    "dog": "🐶", "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
}
THRESHOLD = 0.60

model = CiFAR10()
model.load_state_dict(torch.load(Path(__file__).parent / "best.pt", map_location="cpu"))
model.eval()

prep = T.Compose([T.Resize(32), T.CenterCrop(32)])
to_input = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
footer {display: none !important;}

.vc-hero {padding: 28px 4px 12px;}
.vc-brand {display: flex; align-items: center; gap: 14px;}
.vc-logo {width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center;
          justify-content: center; font-size: 24px;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);}
.vc-hero h1 {font-size: 34px; font-weight: 800; letter-spacing: -0.5px; margin: 0;}
.vc-hero p {margin: 10px 0 14px; opacity: .75; font-size: 16px;}
.vc-chips {display: flex; gap: 8px; flex-wrap: wrap;}
.vc-chip {padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600;
          border: 1px solid var(--border-color-primary);
          background: var(--background-fill-secondary);}

.vc-card {padding: 26px; border-radius: 18px; border: 1px solid var(--border-color-primary);
          background: var(--background-fill-secondary);}
.vc-empty {padding: 48px 24px; border-radius: 18px; text-align: center; opacity: .7;
           border: 1.5px dashed var(--border-color-primary);}
.vc-empty b {display: block; font-size: 18px; margin-bottom: 6px;}

.vc-eyebrow {font-size: 12px; font-weight: 700; letter-spacing: 1.2px;
             text-transform: uppercase; opacity: .55;}
.vc-pred {font-size: 40px; font-weight: 800; text-transform: capitalize; margin: 6px 0 2px;}
.vc-conf {font-size: 15px; opacity: .8;}
.vc-warn {margin-top: 14px; padding: 10px 14px; border-radius: 10px; font-size: 14px;
          background: rgba(245, 158, 11, .12); border: 1px solid rgba(245, 158, 11, .35);}
.vc-divider {height: 1px; background: var(--border-color-primary); margin: 20px 0 14px;}

.vc-bar-row {display: grid; grid-template-columns: 110px 1fr 56px; align-items: center;
             gap: 12px; margin: 11px 0; font-size: 14px;}
.vc-name {text-transform: capitalize;}
.vc-pct {text-align: right; opacity: .8; font-variant-numeric: tabular-nums;}
.vc-track {height: 8px; border-radius: 999px; overflow: hidden;
           background: var(--border-color-primary);}
.vc-fill {height: 100%; border-radius: 999px;
          background: linear-gradient(90deg, #6366f1, #8b5cf6);}
.vc-fill.muted {background: #9ca3af; opacity: .55;}

.vc-note {font-size: 13px; opacity: .65; padding: 4px 2px;}
"""

HEADER = "<style>" + CSS + "</style>" + """
<div class="vc-hero">
  <div class="vc-brand">
    <div class="vc-logo">🧠</div>
    <h1>CIFAR-10 Classifier</h1>
  </div>
  <p>Drop in a photo and a CNN trained from scratch tells you what it sees.</p>
  <div class="vc-chips">
    <span class="vc-chip">91% test accuracy</span>
    <span class="vc-chip">ResNet-9 style CNN</span>
    <span class="vc-chip">PyTorch</span>
    <span class="vc-chip">10 classes</span>
  </div>
</div>
"""

EMPTY = """
<div class="vc-empty">
  <b>No image yet</b>
  Upload, drag and drop, paste, or use your webcam.
</div>
"""

NOTE = (
    '<div class="vc-note">Recognizes: airplane · automobile · bird · cat · deer · dog · '
    'frog · horse · ship · truck. Anything else is forced into one of these, '
    'so use a clear photo with one centered object.</div>'
)


def result_html(probs):
    values, indices = probs.topk(5)
    values, indices = values.tolist(), indices.tolist()
    top_name, conf = CLASSES[indices[0]], values[0]

    warn = ""
    if conf < THRESHOLD:
        warn = '<div class="vc-warn">⚠️ Low confidence. This may not be one of the 10 classes.</div>'

    rows = ""
    for rank, (p, i) in enumerate(zip(values, indices)):
        fill = "vc-fill" if rank == 0 else "vc-fill muted"
        rows += (
            f'<div class="vc-bar-row"><span class="vc-name">{CLASSES[i]}</span>'
            f'<div class="vc-track"><div class="{fill}" style="width:{p * 100:.1f}%"></div></div>'
            f'<span class="vc-pct">{p * 100:.1f}%</span></div>'
        )

    return f"""
    <div class="vc-card">
      <div class="vc-eyebrow">Prediction</div>
      <div class="vc-pred">{EMOJI[top_name]} {top_name}</div>
      <div class="vc-conf">{conf * 100:.1f}% confidence</div>
      {warn}
      <div class="vc-divider"></div>
      <div class="vc-eyebrow">Top 5</div>
      {rows}
    </div>
    """


def predict(img):
    if img is None:
        return EMPTY, None

    small = prep(img.convert("RGB"))
    x = to_input(small).unsqueeze(0)

    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]

    preview = small.resize((256, 256), Image.NEAREST)
    return result_html(probs), preview


with gr.Blocks(title="CIFAR-10 Classifier") as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                label="Your image",
                height=420,
            )
            gr.HTML(NOTE)

        with gr.Column(scale=1):
            result_out = gr.HTML(EMPTY)
            preview_out = gr.Image(
                label="What the model actually sees (32×32)",
                interactive=False,
                height=200,
            )

    inp.change(predict, inp, [result_out, preview_out])


if __name__ == "__main__":
    demo.launch()