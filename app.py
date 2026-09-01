"""
Military Vehicle Detection — Interactive Demo UI
Uses the trained YOLOv8n model to detect 5 classes:
  LMV, SMV, MCV, CV, AFV
"""

import os
import glob
import random
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS  = os.path.join(BASE_DIR, "results", "my_8n_run", "weights", "best.pt")
VAL_DIR  = os.path.join(BASE_DIR, "data", "images", "val")

# ── Load model once at startup ───────────────────────────────────────────────
print(f"Loading model from: {WEIGHTS}")
model = YOLO(WEIGHTS)
print("Model loaded successfully!")

# ── Class info ───────────────────────────────────────────────────────────────
CLASS_INFO = {
    "LMV": {"full": "Light Motor Vehicle",    "color": (46, 204, 113),  "emoji": "🚙"},
    "SMV": {"full": "Small Motor Vehicle",    "color": (52, 152, 219),  "emoji": "🚐"},
    "MCV": {"full": "Military Cargo Vehicle", "color": (243, 156, 18),  "emoji": "🚛"},
    "CV":  {"full": "Combat Vehicle",         "color": (231, 76, 60),   "emoji": "⚔️"},
    "AFV": {"full": "Armored Fighting Vehicle","color": (155, 89, 182), "emoji": "🛡️"},
}

# ── Get sample image paths ───────────────────────────────────────────────────
ALL_VAL_IMAGES = sorted(glob.glob(os.path.join(VAL_DIR, "*.jpg")))
random.seed(42)
SAMPLE_PATHS = random.sample(ALL_VAL_IMAGES, min(12, len(ALL_VAL_IMAGES)))

# ── Detection function ───────────────────────────────────────────────────────
def detect_vehicles(image, confidence_threshold, iou_threshold):
    """Run YOLOv8 inference and return annotated image + detection summary."""
    if image is None:
        return None, "⚠️ Please upload an image first."

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    annotated = image.copy()
    h, w = annotated.shape[:2]

    class_counts = {}
    detections_detail = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        info = CLASS_INFO.get(cls_name, {"full": cls_name, "color": (200,200,200), "emoji": "?"})
        color = info["color"]

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        detections_detail.append(f"  {info['emoji']} **{cls_name}** ({info['full']}) — {conf:.1%}")

        label = f"{cls_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(h, w) / 1200)
        thickness = max(1, int(min(h, w) / 400))
        (tw, th_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness + 1)

        label_y1 = max(y1 - th_text - baseline - 8, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - baseline - 2),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    total = len(result.boxes)
    if total == 0:
        summary = "### No vehicles detected\nTry lowering the confidence threshold."
    else:
        summary = f"### 🎯 {total} Vehicle{'s' if total > 1 else ''} Detected\n\n"
        summary += "| Class | Count |\n|-------|-------|\n"
        for cls_name in ["LMV", "SMV", "MCV", "CV", "AFV"]:
            if cls_name in class_counts:
                info = CLASS_INFO[cls_name]
                summary += f"| {info['emoji']} {cls_name} ({info['full']}) | {class_counts[cls_name]} |\n"
        summary += f"\n---\n**Detection details:**\n\n"
        summary += "\n".join(detections_detail)

    return annotated, summary


def load_random_sample():
    """Load a random image from the validation set and run detection."""
    if not ALL_VAL_IMAGES:
        return None
    chosen = random.choice(ALL_VAL_IMAGES)
    img = cv2.imread(chosen)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ── Build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Military Vehicle Detection — YOLOv8",
) as demo:

    gr.HTML("""
    <div style="text-align:center; background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                color:white; padding:24px; border-radius:12px; margin-bottom:16px;">
        <h1 style="margin:0; font-size:2em; letter-spacing:1px;">🎖️ Military Vehicle Detection System</h1>
        <p style="margin:8px 0 0 0; opacity:0.85; font-size:1.05em;">
            YOLOv8n • MV-RSD Dataset • 5 Classes: LMV / SMV / MCV / CV / AFV
        </p>
        <p style="font-size:0.85em; margin-top:12px; opacity:0.7;">
            Model: best.pt (epoch 47) &nbsp;|&nbsp; mAP50: 85.9% &nbsp;|&nbsp; mAP50-95: 59.2% &nbsp;|&nbsp; ~149 FPS
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Input Image",
                type="numpy",
                height=420,
            )

            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.25, step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence to show a detection"
                )
                iou_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.45, step=0.05,
                    label="IoU Threshold (NMS)",
                    info="Non-max suppression overlap threshold"
                )

            with gr.Row():
                detect_btn = gr.Button("🔍 Detect Vehicles", variant="primary", size="lg")
                random_btn = gr.Button("🎲 Random Val Image", variant="secondary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🎯 Detection Result",
                type="numpy",
                height=420,
            )
            detection_info = gr.Markdown(
                value="### Upload an image and click **Detect Vehicles** to begin.",
            )

    gr.Markdown("### 📂 Sample Images from Validation Set\nClick any image to load it for detection.")
    sample_gallery = gr.Gallery(
        value=SAMPLE_PATHS,
        label="Validation Samples",
        columns=6,
        rows=2,
        height=200,
        object_fit="cover",
        allow_preview=False,
    )

    with gr.Accordion("📊 Model Performance Summary", open=False):
        gr.Markdown("""
| Metric | Overall | LMV | SMV | MCV | CV | AFV |
|--------|---------|-----|-----|-----|----|----|
| **Precision** | 0.837 | 0.891 | 0.854 | 0.853 | 0.944 | 0.644 |
| **Recall** | 0.804 | 0.765 | 0.833 | 0.903 | 0.940 | 0.580 |
| **mAP50** | 0.859 | 0.887 | 0.900 | 0.935 | 0.977 | 0.597 |
| **mAP50-95** | 0.592 | 0.614 | 0.624 | 0.637 | 0.707 | 0.379 |

**Architecture:** YOLOv8n (3.0M params, 8.1 GFLOPs) — trained for 50 epochs on MV-RSD
**Best checkpoint:** Epoch 47 | **Inference speed:** ~5.2ms/image on RTX 3050
        """)

    # ── Events ──
    detect_btn.click(
        fn=detect_vehicles,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, detection_info],
    )

    random_btn.click(
        fn=load_random_sample,
        inputs=[],
        outputs=[input_image],
    ).then(
        fn=detect_vehicles,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, detection_info],
    )

    def on_gallery_select(evt: gr.SelectData):
        """Load clicked gallery image."""
        val = evt.value
        # Gradio 6 gallery returns dict with image info
        if isinstance(val, dict):
            path = val.get("image", {}).get("path", "") if isinstance(val.get("image"), dict) else val.get("url", val.get("path", ""))
        elif isinstance(val, str):
            path = val
        else:
            return None
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    sample_gallery.select(
        fn=on_gallery_select,
        inputs=[],
        outputs=[input_image],
    )


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Military Vehicle Detection — Demo UI")
    print("  Open http://localhost:7860 in your browser")
    print("="*60 + "\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
