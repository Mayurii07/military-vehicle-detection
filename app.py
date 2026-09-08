"""
Military Vehicle Detection — Interactive Demo UI
Uses the trained YOLOv8n model to detect 5 classes:
  LMV, SMV, MCV, CV, AFV
"""

import os
import glob
import random
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_CANDIDATES = (
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "results", "my_8n_run", "weights", "best.pt"),
)
WEIGHTS = next((path for path in WEIGHT_CANDIDATES if os.path.isfile(path)), WEIGHT_CANDIDATES[0])
VAL_DIR  = os.path.join(BASE_DIR, "data", "images", "val")

@st.cache_resource
def load_model():
    """Load the detector once per Streamlit process."""
    if not os.path.isfile(WEIGHTS):
        raise FileNotFoundError(
            "Model weights were not found. Place best.pt in the project root "
            "or train the model so it is saved under results/my_8n_run/weights/."
        )
    return YOLO(WEIGHTS)

# ── Class info ───────────────────────────────────────────────────────────────
CLASS_INFO = {
    "LMV": {"full": "Light Motor Vehicle",    "color": (46, 204, 113),  "emoji": "🚙"},
    "SMV": {"full": "Small Motor Vehicle",    "color": (52, 152, 219),  "emoji": "🚐"},
    "MCV": {"full": "Military Cargo Vehicle", "color": (243, 156, 18),  "emoji": "🚛"},
    "CV":  {"full": "Combat Vehicle",         "color": (231, 76, 60),   "emoji": "⚔️"},
    "AFV": {"full": "Armored Fighting Vehicle","color": (155, 89, 182), "emoji": "🛡️"},
}

# ── Get sample image paths ───────────────────────────────────────────────────
IMAGE_ROOTS = (
    os.path.join(BASE_DIR, "data", "images"),
    os.path.join(BASE_DIR, "data", "test"),
    os.path.join(BASE_DIR, "MVRSD_dataset"),
)
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_sample_images():
    """Collect all project images while ignoring duplicate dataset copies."""
    images_by_name = {}
    for root in IMAGE_ROOTS:
        if not os.path.isdir(root):
            continue
        for extension in IMAGE_EXTENSIONS:
            for path in glob.glob(os.path.join(root, "**", extension), recursive=True):
                key = os.path.basename(path).lower()
                images_by_name.setdefault(key, path)
    return sorted(images_by_name.values(), key=lambda path: os.path.basename(path).lower())


ALL_SAMPLE_IMAGES = collect_sample_images()
SAMPLE_LABELS = {
    path: f"{os.path.basename(path)} ({os.path.relpath(path, BASE_DIR)})"
    for path in ALL_SAMPLE_IMAGES
}
SAMPLE_PATH_BY_LABEL = {label: path for path, label in SAMPLE_LABELS.items()}
SAMPLE_OPTIONS = ["Select a sample image"] + list(SAMPLE_PATH_BY_LABEL)

# ── Detection function ───────────────────────────────────────────────────────
def detect_vehicles(image, confidence_threshold, iou_threshold, detector):
    """Run YOLOv8 inference and return annotated image + detection summary."""
    if image is None:
        return None, "⚠️ Please upload an image first."

    results = detector.predict(
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
    """Load a random image from the complete sample pool."""
    previous_path = st.session_state.get("random_sample_path")
    candidates = [path for path in ALL_SAMPLE_IMAGES if path != previous_path]
    if not candidates:
        candidates = ALL_SAMPLE_IMAGES
    if not candidates:
        return None

    chosen = random.choice(candidates)
    st.session_state["random_sample_path"] = chosen
    img = cv2.imread(chosen)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


st.set_page_config(page_title="Military Vehicle Detection", page_icon="🎖️", layout="wide")

st.session_state.setdefault("inference_history", [])

st.markdown("""
<style>
.hero { background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        color: white; padding: 24px; border-radius: 12px; text-align: center; }
</style>
<div class="hero">
<h1>🎖️ Military Vehicle Detection System</h1>
<p>YOLOv8n • MV-RSD Dataset • 5 Classes: LMV / SMV / MCV / CV / AFV</p>
<p>Model: best.pt (epoch 47) | mAP50: 85.9% | mAP50-95: 59.2%</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.subheader("Detect vehicles")
left, right = st.columns(2)
with left:
    uploaded_file = st.file_uploader("📸 Input Image", type=["jpg", "jpeg", "png"])
    with st.expander("📹 Webcam Feed"):
        webcam_frame = st.camera_input("Capture a vehicle image")
        webcam_detect_clicked = st.button(
            "🔍 Detect Webcam Frame",
            use_container_width=True,
            disabled=webcam_frame is None,
        )

    selected_sample = st.selectbox(
        "📂 Sample Image",
        SAMPLE_OPTIONS,
        index=SAMPLE_OPTIONS.index(st.session_state.get("selected_sample", SAMPLE_OPTIONS[0]))
        if st.session_state.get("selected_sample", SAMPLE_OPTIONS[0]) in SAMPLE_OPTIONS
        else 0,
    )
    st.session_state["selected_sample"] = selected_sample
    detect_clicked = st.button("🔍 Detect Vehicles", type="primary", use_container_width=True)
    random_clicked = st.button("🎲 Random Val Image", use_container_width=True)

if random_clicked:
    st.session_state["selected_sample"] = SAMPLE_OPTIONS[0]

with st.sidebar:
    st.header("⚙️ Settings")
    display_theme = st.radio("Display Theme", ["Dark", "Light", "System"], index=0)
    language = st.selectbox("Language", ["English"], index=0)
    st.checkbox(
        "Progressive Web App",
        value=False,
        disabled=True,
        help="Gradio PWA installation is not available in Streamlit.",
    )
    st.checkbox(
        "Screen Studio",
        value=False,
        disabled=True,
        help="Screen Studio is a Gradio feature and is not available in Streamlit.",
    )
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.1, 0.95, 0.25, 0.05,
        help="Minimum confidence to show a detection",
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)", 0.1, 0.95, 0.45, 0.05,
        help="Non-max suppression overlap threshold",
    )
    st.caption(f"Language: {language} | Theme: {display_theme}")

    st.subheader("Run History")
    if st.button("Clear history", use_container_width=True):
        st.session_state["inference_history"] = []
        st.rerun()
    st.caption(f"{len(st.session_state['inference_history'])} inference(s) saved in this browser session")

with right:
    st.subheader("🎯 Detection Result")
    output_placeholder = st.empty()
    info_placeholder = st.empty()

image = None
if random_clicked:
    image = load_random_sample()
elif uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
elif selected_sample != SAMPLE_OPTIONS[0]:
    image = cv2.imread(SAMPLE_PATH_BY_LABEL[selected_sample])
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is None and st.session_state.get("random_sample_path"):
    image = cv2.imread(st.session_state["random_sample_path"])
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if random_clicked and image is not None:
    st.session_state["selected_image"] = image
elif image is not None:
    st.session_state["selected_image"] = image
else:
    image = st.session_state.get("selected_image")

if image is not None:
    with left:
        st.image(image, caption="Input image", use_container_width=True)

if detect_clicked or random_clicked:
    if image is None:
        info_placeholder.warning("Please upload an image or choose a validation sample first.")
    else:
        try:
            annotated, summary = detect_vehicles(
                image, confidence_threshold, iou_threshold, model
            )
            st.session_state["annotated_image"] = annotated
            st.session_state["detection_summary"] = summary
            st.session_state["inference_history"].insert(
                0,
                {
                    "input": image.copy(),
                    "output": annotated.copy(),
                    "confidence": confidence_threshold,
                    "iou": iou_threshold,
                    "summary": summary,
                },
            )
            st.session_state["inference_history"] = st.session_state["inference_history"][:100]
        except Exception as error:
            info_placeholder.error(f"Detection failed: {error}")

if webcam_detect_clicked:
    if webcam_frame is None:
        info_placeholder.warning("Allow camera access and capture a frame first.")
    else:
        try:
            webcam_bytes = webcam_frame.getvalue()
            webcam_image = cv2.imdecode(
                np.frombuffer(webcam_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if webcam_image is None:
                raise ValueError("The webcam frame could not be decoded as an image.")
            webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
            annotated, summary = detect_vehicles(
                webcam_image, confidence_threshold, iou_threshold, model
            )
            st.session_state["annotated_image"] = annotated
            st.session_state["detection_summary"] = summary
            st.session_state["inference_history"].insert(
                0,
                {
                    "input": webcam_image.copy(),
                    "output": annotated.copy(),
                    "confidence": confidence_threshold,
                    "iou": iou_threshold,
                    "summary": summary,
                },
            )
            st.session_state["inference_history"] = st.session_state["inference_history"][:100]
        except Exception as error:
            info_placeholder.error(f"Webcam detection failed: {error}")

if "annotated_image" in st.session_state:
    output_placeholder.image(
        st.session_state["annotated_image"],
        caption="Detected vehicles",
        width="stretch",
    )
    info_placeholder.markdown(st.session_state["detection_summary"])

with st.expander("📊 Model Performance Summary"):
    st.markdown("""
| Metric | Overall | LMV | SMV | MCV | CV | AFV |
|--------|---------|-----|-----|-----|----|----|
| **Precision** | 0.837 | 0.891 | 0.854 | 0.853 | 0.944 | 0.644 |
| **Recall** | 0.804 | 0.765 | 0.833 | 0.903 | 0.940 | 0.580 |
| **mAP50** | 0.859 | 0.887 | 0.900 | 0.935 | 0.977 | 0.597 |
| **mAP50-95** | 0.592 | 0.614 | 0.624 | 0.637 | 0.707 | 0.379 |

**Architecture:** YOLOv8n (3.0M params, 8.1 GFLOPs) — trained for 50 epochs on MV-RSD
**Best checkpoint:** Epoch 47 | **Inference speed:** ~5.2ms/image on RTX 3050
    """)

with st.expander(f"🧾 Run History ({len(st.session_state['inference_history'])})"):
    if not st.session_state["inference_history"]:
        st.info("No completed detections in this browser session.")
    else:
        for index, entry in enumerate(st.session_state["inference_history"], start=1):
            st.markdown(f"**Run {index}** | Confidence: `{entry['confidence']:.2f}` | IoU: `{entry['iou']:.2f}`")
            history_input, history_output = st.columns(2)
            with history_input:
                st.image(entry["input"], caption="Input image", use_container_width=True)
            with history_output:
                st.image(entry["output"], caption="Detection result", use_container_width=True)
            st.markdown(entry["summary"])
            if index < len(st.session_state["inference_history"]):
                st.divider()
