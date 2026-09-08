"""
Military Vehicle Detection — Interactive Demo UI
Uses the trained YOLOv8n model to detect 5 classes:
  LMV, SMV, MCV, CV, AFV
"""

import os
import glob
import random
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
import torch
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

TRANSLATIONS = {
    "English": {
        "settings": "Settings", "theme": "Display Theme", "language": "Language",
        "confidence": "Confidence Threshold", "iou": "IoU Threshold (NMS)",
        "image_size": "Inference Image Size", "max_detections": "Maximum Detections",
        "classes": "Vehicle Classes", "tta": "Test-Time Augmentation",
        "device": "Inference Device", "history_limit": "Maximum History Runs",
        "reset": "Reset to Defaults", "download": "Download Result",
        "run_history": "Run History", "clear_history": "Clear history",
        "detect": "Detect Vehicles", "random": "Random Val Image",
        "sample": "Sample Image", "input": "Input Image", "result": "Detection Result",
    },
    "Hindi": {
        "settings": "सेटिंग्स", "theme": "डिस्प्ले थीम", "language": "भाषा",
        "confidence": "विश्वास सीमा", "iou": "IoU सीमा (NMS)",
        "image_size": "इन्फरेंस इमेज आकार", "max_detections": "अधिकतम पहचान",
        "classes": "वाहन वर्ग", "tta": "टेस्ट-टाइम ऑगमेंटेशन",
        "device": "इन्फरेंस डिवाइस", "history_limit": "अधिकतम इतिहास रन",
        "reset": "डिफ़ॉल्ट पर रीसेट करें", "download": "परिणाम डाउनलोड करें",
        "run_history": "रन इतिहास", "clear_history": "इतिहास साफ़ करें",
        "detect": "वाहन पहचानें", "random": "रैंडम वैलिडेशन इमेज",
        "sample": "सैंपल इमेज", "input": "इनपुट इमेज", "result": "पहचान परिणाम",
    },
    "Spanish": {
        "settings": "Configuración", "theme": "Tema de pantalla", "language": "Idioma",
        "confidence": "Umbral de confianza", "iou": "Umbral IoU (NMS)",
        "image_size": "Tamaño de inferencia", "max_detections": "Detecciones máximas",
        "classes": "Clases de vehículos", "tta": "Aumento en tiempo de prueba",
        "device": "Dispositivo de inferencia", "history_limit": "Máximo de ejecuciones",
        "reset": "Restablecer valores", "download": "Descargar resultado",
        "run_history": "Historial de ejecuciones", "clear_history": "Borrar historial",
        "detect": "Detectar vehículos", "random": "Imagen de validación aleatoria",
        "sample": "Imagen de muestra", "input": "Imagen de entrada", "result": "Resultado",
    },
    "French": {
        "settings": "Paramètres", "theme": "Thème d'affichage", "language": "Langue",
        "confidence": "Seuil de confiance", "iou": "Seuil IoU (NMS)",
        "image_size": "Taille d'inférence", "max_detections": "Détections maximales",
        "classes": "Classes de véhicules", "tta": "Augmentation au test",
        "device": "Appareil d'inférence", "history_limit": "Exécutions historiques maximales",
        "reset": "Réinitialiser", "download": "Télécharger le résultat",
        "run_history": "Historique des exécutions", "clear_history": "Effacer l'historique",
        "detect": "Détecter les véhicules", "random": "Image de validation aléatoire",
        "sample": "Image échantillon", "input": "Image d'entrée", "result": "Résultat",
    },
    "German": {
        "settings": "Einstellungen", "theme": "Darstellung", "language": "Sprache",
        "confidence": "Konfidenzschwelle", "iou": "IoU-Schwelle (NMS)",
        "image_size": "Inferenzbildgröße", "max_detections": "Maximale Erkennungen",
        "classes": "Fahrzeugklassen", "tta": "Testzeit-Augmentierung",
        "device": "Inferenzgerät", "history_limit": "Maximale Verlaufsläufe",
        "reset": "Standards wiederherstellen", "download": "Ergebnis herunterladen",
        "run_history": "Verlauf", "clear_history": "Verlauf löschen",
        "detect": "Fahrzeuge erkennen", "random": "Zufälliges Validierungsbild",
        "sample": "Beispielbild", "input": "Eingabebild", "result": "Ergebnis",
    },
    "Japanese": {
        "settings": "設定", "theme": "表示テーマ", "language": "言語",
        "confidence": "信頼度しきい値", "iou": "IoUしきい値 (NMS)",
        "image_size": "推論画像サイズ", "max_detections": "最大検出数",
        "classes": "車両クラス", "tta": "テスト時拡張",
        "device": "推論デバイス", "history_limit": "履歴の最大実行数",
        "reset": "デフォルトに戻す", "download": "結果をダウンロード",
        "run_history": "実行履歴", "clear_history": "履歴を消去",
        "detect": "車両を検出", "random": "ランダム検証画像",
        "sample": "サンプル画像", "input": "入力画像", "result": "検出結果",
    },
    "Chinese": {
        "settings": "设置", "theme": "显示主题", "language": "语言",
        "confidence": "置信度阈值", "iou": "IoU 阈值 (NMS)",
        "image_size": "推理图像大小", "max_detections": "最大检测数",
        "classes": "车辆类别", "tta": "测试时增强",
        "device": "推理设备", "history_limit": "最大历史运行数",
        "reset": "恢复默认设置", "download": "下载结果",
        "run_history": "运行历史", "clear_history": "清除历史",
        "detect": "检测车辆", "random": "随机验证图像",
        "sample": "示例图像", "input": "输入图像", "result": "检测结果",
    },
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
def resolve_device(device):
    """Map the UI device choices to a device accepted by Ultralytics."""
    requested_device = (device or "auto").strip().lower()
    if requested_device in ("", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def detect_vehicles(
    image,
    confidence_threshold,
    iou_threshold,
    detector,
    image_size=640,
    max_detections=300,
    selected_classes=None,
    use_tta=False,
    device="auto",
):
    """Run YOLOv8 inference and return annotated image + detection summary."""
    if image is None:
        return None, "⚠️ Please upload an image first."

    results = detector.predict(
        source=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        max_det=max_detections,
        classes=selected_classes,
        augment=use_tta,
        device=resolve_device(device),
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
        detections_detail.append(
            f"| {info['emoji']} {cls_name} | {info['full']} | {conf:.1%} |"
        )

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
        summary += "\n---\n**Detection details:**\n\n"
        summary += "| Class | Vehicle Type | Confidence |\n"
        summary += "|:--|:--|--:|\n"
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
st.session_state.setdefault("selected_image", None)
st.session_state.setdefault("selected_image_source", None)
st.session_state.setdefault("loaded_run_id", None)
st.session_state.setdefault("language", "English")
st.session_state.setdefault("history_limit", 100)
st.session_state.setdefault("device", "Auto")
labels = TRANSLATIONS[st.session_state["language"]]

pending_load = st.session_state.pop("pending_load", None)
if pending_load is not None:
    st.session_state["selected_image"] = pending_load["input"]
    st.session_state["selected_image_source"] = pending_load.get("source", "Image")
    st.session_state["annotated_image"] = pending_load["output"]
    st.session_state["detection_summary"] = pending_load["summary"]
    st.session_state["confidence_threshold"] = pending_load["confidence"]
    st.session_state["iou_threshold"] = pending_load["iou"]
    st.session_state["image_size"] = pending_load.get("image_size", 640)
    st.session_state["max_detections"] = pending_load.get("max_detections", 300)
    st.session_state["selected_classes"] = pending_load.get("selected_classes", list(CLASS_INFO))
    st.session_state["use_tta"] = pending_load.get("use_tta", False)
    st.session_state["device"] = pending_load.get("device", "Auto")
    st.session_state["loaded_run_id"] = pending_load["id"]

st.markdown("""
<style>
.hero { background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    color: #ffffff !important; padding: 24px; border-radius: 12px; text-align: center; }
.hero h1, .hero p { color: #ffffff !important; }
</style>
<div class="hero">
<h1>🎖️ Military Vehicle Detection System</h1>
<p>YOLOv8n • MV-RSD Dataset • 5 Classes: LMV / SMV / MCV / CV / AFV</p>
<p>Model: best.pt (epoch 47) | mAP50: 85.9% | mAP50-95: 59.2%</p>
</div>
""", unsafe_allow_html=True)

theme_css = {
    "Dark": """
        <style>
        :root { color-scheme: dark; }
        :root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            --app-background: #0f1117;
            --app-surface: #20232c;
            --app-surface-muted: #171a21;
            --app-text: #f4f4f4;
            --app-text-muted: #c8ccd6;
            --app-border: #454b59;
            --app-accent: #ff4b4b;
            --app-focus: #70a7ff;
            background: var(--app-background);
            color: var(--app-text);
        }
        [data-testid="stSidebar"] {
            background: var(--app-surface-muted);
            color: var(--app-text);
        }
        </style>
    """,
    "Light": """
        <style>
        :root { color-scheme: light; }
        :root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            --app-background: #f7f8fb;
            --app-surface: #ffffff;
            --app-surface-muted: #eef1f6;
            --app-text: #172033;
            --app-text-muted: #4b5568;
            --app-border: #b8c0ce;
            --app-accent: #d9363e;
            --app-focus: #1769aa;
            background: var(--app-background);
            color: var(--app-text);
        }
        [data-testid="stSidebar"] {
            background: var(--app-surface-muted);
            color: var(--app-text);
        }
        </style>
    """,
    "System": """
        <style>
        @media (prefers-color-scheme: dark) {
            :root { color-scheme: dark; }
            :root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
                --app-background: #0f1117;
                --app-surface: #20232c;
                --app-surface-muted: #171a21;
                --app-text: #f4f4f4;
                --app-text-muted: #c8ccd6;
                --app-border: #454b59;
                --app-accent: #ff4b4b;
                --app-focus: #70a7ff;
            }
        }
        @media (prefers-color-scheme: light) {
            :root { color-scheme: light; }
            :root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
                --app-background: #f7f8fb;
                --app-surface: #ffffff;
                --app-surface-muted: #eef1f6;
                --app-text: #172033;
                --app-text-muted: #4b5568;
                --app-border: #b8c0ce;
                --app-accent: #d9363e;
                --app-focus: #1769aa;
            }
        }
        [data-testid="stAppViewContainer"] {
            background: var(--app-background);
            color: var(--app-text);
        }
        [data-testid="stSidebar"] {
            background: var(--app-surface-muted);
            color: var(--app-text);
        }
        </style>
    """,
}

theme_components_css = """
<style>
:root {
    --app-on-accent: #ffffff;
}
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    color: var(--app-text);
}
.hero, .hero h1, .hero p {
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--app-text);
}
[data-testid="stAppViewContainer"] small,
[data-testid="stSidebar"] small,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--app-text-muted);
}
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stCameraInput"] {
    background: var(--app-surface);
    color: var(--app-text);
    border-color: var(--app-border);
}
[data-baseweb="select"] [role="option"],
[data-baseweb="popover"] [role="listbox"] {
    background: var(--app-surface);
    color: var(--app-text);
}
[data-testid="stAppViewContainer"] button,
[data-testid="stSidebar"] button {
    background: var(--app-surface);
    color: var(--app-text);
    border-color: var(--app-border);
}
[data-testid="stAppViewContainer"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stExpander"] summary:hover {
    background: var(--app-surface-muted);
    border-color: var(--app-focus);
    color: var(--app-text);
}
[data-testid="stAppViewContainer"] button[kind="primary"],
[data-testid="stAppViewContainer"] [data-testid="baseButton-primary"] {
    color: var(--app-on-accent) !important;
    background: var(--app-accent) !important;
    border-color: var(--app-accent) !important;
}
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] button {
    color: var(--app-text) !important;
    background: var(--app-surface) !important;
    border-color: var(--app-border) !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] section,
[data-testid="stFileUploaderDropzone"] svg {
    color: var(--app-text-muted) !important;
    fill: currentColor;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary svg {
    color: var(--app-text) !important;
    fill: currentColor;
}
[data-testid="stSlider"] label,
[data-testid="stSlider"] [data-testid="stMarkdownContainer"],
[data-testid="stSlider"] output,
[data-testid="stSlider"] svg {
    color: var(--app-text) !important;
    fill: currentColor;
}
[data-testid="stTooltipIcon"],
[data-testid="stTooltipIcon"] svg,
[data-testid="stHelp"] svg,
[data-testid="stWidgetLabel"] svg {
    color: var(--app-text-muted) !important;
    fill: currentColor !important;
    opacity: 1 !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: var(--app-background) !important;
    color: var(--app-text) !important;
}
[data-testid="stHeader"] button, [data-testid="stToolbar"] button,
[data-testid="stHeader"] a, [data-testid="stToolbar"] a {
    color: var(--app-text) !important;
    fill: currentColor !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    color: var(--app-text) !important;
    background: var(--app-surface) !important;
    border-color: var(--app-border) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stMultiSelect"] [data-baseweb="select"] svg {
    color: var(--app-text) !important;
    fill: currentColor !important;
}
[data-baseweb="tag"] {
    color: var(--app-on-accent) !important;
    background: var(--app-accent) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] > div,
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: var(--app-on-accent) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] button,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:hover,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:focus {
    min-width: 1.25rem;
    width: 1.25rem;
    height: 1.25rem;
    padding: 0;
    color: var(--app-on-accent) !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] button *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button *:hover {
    color: var(--app-on-accent) !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}
[data-baseweb="tag"] svg {
    color: var(--app-on-accent) !important;
    fill: currentColor !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    background: var(--app-surface) !important;
    border-color: var(--app-border) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: var(--app-surface) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child button,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child button:hover {
    color: var(--app-text) !important;
    background: var(--app-surface-muted) !important;
    border-color: var(--app-border) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child svg {
    color: var(--app-text) !important;
    fill: currentColor !important;
}
[data-testid="stMultiSelect"] button[aria-label="Clear all"],
[data-testid="stMultiSelect"] button[aria-label="Clear all"]:hover {
    color: var(--app-text) !important;
    background: var(--app-surface-muted) !important;
    border-color: var(--app-border) !important;
}
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label svg {
    color: var(--app-text) !important;
    fill: currentColor;
}
[data-testid="stCheckbox"] input:disabled + div {
    border-color: var(--app-border) !important;
    background: var(--app-surface) !important;
}
[data-testid="stMultiSelect"] {
    --vehicle-input: var(--app-surface);
    --vehicle-control: var(--app-surface-muted);
    --vehicle-text: var(--app-text);
    --vehicle-border: var(--app-border);
    --vehicle-accent: var(--app-accent);
}
[data-testid="stMultiSelect"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: var(--vehicle-input) !important;
    background-color: var(--vehicle-input) !important;
    border-color: var(--vehicle-border) !important;
    color: var(--vehicle-text) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div {
    background: transparent !important;
    color: var(--vehicle-text) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] input,
[data-testid="stMultiSelect"] [data-baseweb="select"] input::placeholder {
    background: transparent !important;
    color: var(--vehicle-text) !important;
    caret-color: var(--vehicle-text);
    opacity: 1;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child button {
    background: var(--vehicle-control) !important;
    background-color: var(--vehicle-control) !important;
    border-color: var(--vehicle-border) !important;
    color: var(--vehicle-text) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child svg,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child path {
    color: var(--vehicle-text) !important;
    fill: currentColor !important;
    stroke: currentColor !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.35rem !important;
    min-width: 3.6rem !important;
    margin: 0.2rem 0.25rem 0.2rem 0 !important;
    padding: 0.35rem 0.55rem 0.35rem 0.7rem !important;
    background: var(--vehicle-accent) !important;
    background-color: var(--vehicle-accent) !important;
    color: var(--app-on-accent) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: var(--app-on-accent) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] > span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] > div:first-child {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    overflow: visible !important;
    white-space: nowrap !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] button,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:hover,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:focus,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button svg,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button path {
    flex: 0 0 1.35rem !important;
    min-width: 1.35rem !important;
    width: 1.35rem !important;
    height: 1.35rem !important;
    margin: 0 !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 999px !important;
    background: transparent !important;
    background-color: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--app-on-accent) !important;
    fill: currentColor !important;
    stroke: currentColor !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:hover,
[data-testid="stMultiSelect"] [data-baseweb="tag"] button:focus-visible {
    background: rgba(255, 255, 255, 0.2) !important;
    outline: 2px solid rgba(255, 255, 255, 0.85) !important;
    outline-offset: 1px;
}
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
    border-color: var(--app-focus) !important;
    box-shadow: 0 0 0 1px var(--app-focus) !important;
}
[data-testid="stMultiSelect"] button[aria-label="Clear all"],
[data-testid="stMultiSelect"] button[aria-label="Clear all"]:hover,
[data-testid="stMultiSelect"] button[aria-label="Clear all"]:focus {
    background: var(--vehicle-control) !important;
    background-color: var(--vehicle-control) !important;
    border-color: var(--vehicle-border) !important;
    color: var(--vehicle-text) !important;
}
[data-testid="stMultiSelect"] button[aria-label="Clear all"] svg,
[data-testid="stMultiSelect"] button[aria-label="Clear all"] path {
    color: var(--vehicle-text) !important;
    fill: currentColor !important;
    stroke: currentColor !important;
}
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] input:focus) [data-baseweb="popover"],
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] [aria-expanded="true"]) [data-baseweb="popover"] {
    background: var(--vehicle-input) !important;
    border-color: var(--vehicle-border) !important;
}
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] input:focus) [data-baseweb="popover"] [role="option"],
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] [aria-expanded="true"]) [data-baseweb="popover"] [role="option"] {
    background: var(--vehicle-input) !important;
    color: var(--vehicle-text) !important;
}
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] input:focus) [data-baseweb="popover"] [role="option"]:hover,
body:has([data-testid="stMultiSelect"] [data-baseweb="select"] [aria-expanded="true"]) [data-baseweb="popover"] [role="option"]:hover {
    background: var(--vehicle-control) !important;
    color: var(--vehicle-text) !important;
}
[data-testid="stAppViewContainer"] button:disabled,
[data-testid="stSidebar"] button:disabled {
    color: var(--app-text-muted);
    opacity: 0.65;
}
[data-testid="stAppViewContainer"] input::placeholder,
[data-testid="stSidebar"] input::placeholder {
    color: var(--app-text-muted);
    opacity: 1;
}
[data-testid="stAppViewContainer"] input:focus,
[data-testid="stAppViewContainer"] textarea:focus,
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--app-focus) !important;
    box-shadow: 0 0 0 1px var(--app-focus) !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] {
    color: var(--app-text);
}
[data-testid="stAppViewContainer"] hr,
[data-testid="stSidebar"] hr {
    border-color: var(--app-border);
}
</style>
"""

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
        f"📂 {labels['sample']}",
        SAMPLE_OPTIONS,
        index=SAMPLE_OPTIONS.index(st.session_state.get("selected_sample", SAMPLE_OPTIONS[0]))
        if st.session_state.get("selected_sample", SAMPLE_OPTIONS[0]) in SAMPLE_OPTIONS
        else 0,
    )
    st.session_state["selected_sample"] = selected_sample
    detect_clicked = st.button(f"🔍 {labels['detect']}", type="primary", use_container_width=True)
    random_clicked = st.button(f"🎲 {labels['random']}", use_container_width=True)

if random_clicked:
    st.session_state["selected_sample"] = SAMPLE_OPTIONS[0]
    st.session_state["loaded_run_id"] = None

with st.sidebar:
    st.header(f"⚙️ {labels['settings']}")
    display_theme = st.radio("Display Theme", ["Dark", "Light", "System"], index=0, key="display_theme")
    language = st.selectbox(
        labels["language"],
        ["English", "Hindi", "Spanish", "French", "German", "Japanese", "Chinese"],
        index=0,
        key="language",
    )
    labels = TRANSLATIONS[language]
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
        labels["confidence"], 0.1, 0.95, 0.25, 0.05,
        help="Minimum confidence to show a detection",
        key="confidence_threshold",
    )
    iou_threshold = st.slider(
        labels["iou"], 0.1, 0.95, 0.45, 0.05,
        help="Non-max suppression overlap threshold",
        key="iou_threshold",
    )
    image_size = st.select_slider(
        labels["image_size"], options=[320, 480, 640, 800, 1024], value=640, key="image_size"
    )
    max_detections = st.slider(labels["max_detections"], 1, 300, 300, 1, key="max_detections")
    with st.container(border=True):
        selected_classes = st.pills(
            labels["classes"],
            list(CLASS_INFO),
            selection_mode="multi",
            default=list(CLASS_INFO),
            key="selected_classes",
            help="Only selected vehicle classes are returned by the model.",
            label_visibility="visible",
        )
    use_tta = st.checkbox(
        labels["tta"], value=False, key="use_tta",
        help="Run augmented inference for potentially better accuracy at extra cost.",
    )
    device_options = ["Auto", "CPU"] + (["CUDA"] if torch.cuda.is_available() else [])
    device = st.selectbox(labels["device"], device_options, key="device")
    history_limit = st.slider(labels["history_limit"], 1, 100, 100, 1, key="history_limit")
    if st.button(labels["reset"], use_container_width=True):
        for key, value in {
            "display_theme": "Dark", "language": "English", "confidence_threshold": 0.25,
            "iou_threshold": 0.45, "image_size": 640, "max_detections": 300,
            "selected_classes": list(CLASS_INFO), "use_tta": False, "device": "Auto",
            "history_limit": 100,
        }.items():
            st.session_state[key] = value
        st.rerun()

    st.caption(f"Language: {language} | Theme: {display_theme}")
    st.markdown(theme_css[display_theme] + theme_components_css, unsafe_allow_html=True)

    st.subheader(labels["run_history"])
    if st.button(labels["clear_history"], use_container_width=True):
        st.session_state["inference_history"] = []
        st.rerun()
    st.caption(f"{len(st.session_state['inference_history'])} inference(s) saved in this browser session")

selected_class_ids = [list(CLASS_INFO).index(name) for name in selected_classes]

with right:
    st.subheader("🎯 Detection Result")
    output_placeholder = st.empty()
    info_placeholder = st.empty()

image = None
if st.session_state.get("loaded_run_id") is not None:
    image = st.session_state.get("selected_image")
elif random_clicked:
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
    elif not selected_class_ids:
        info_placeholder.warning("Select at least one vehicle class in Settings.")
    else:
        try:
            annotated, summary = detect_vehicles(
                image, confidence_threshold, iou_threshold, model, image_size, max_detections,
                selected_class_ids, use_tta, device.lower()
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
                    "image_size": image_size,
                    "max_detections": max_detections,
                    "selected_classes": selected_classes,
                    "use_tta": use_tta,
                    "device": device,
                    "summary": summary,
                    "timestamp": datetime.now().astimezone(),
                    "source": "Image",
                },
            )
            st.session_state["inference_history"] = st.session_state["inference_history"][:history_limit]
        except Exception as error:
            info_placeholder.error(f"Detection failed: {error}")

if webcam_detect_clicked:
    if webcam_frame is None:
        info_placeholder.warning("Allow camera access and capture a frame first.")
    elif not selected_class_ids:
        info_placeholder.warning("Select at least one vehicle class in Settings.")
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
                webcam_image, confidence_threshold, iou_threshold, model, image_size, max_detections,
                selected_class_ids, use_tta, device.lower()
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
                    "image_size": image_size,
                    "max_detections": max_detections,
                    "selected_classes": selected_classes,
                    "use_tta": use_tta,
                    "device": device,
                    "summary": summary,
                    "timestamp": datetime.now().astimezone(),
                    "source": "Webcam",
                },
            )
            st.session_state["inference_history"] = st.session_state["inference_history"][:history_limit]
        except Exception as error:
            info_placeholder.error(f"Webcam detection failed: {error}")

if "annotated_image" in st.session_state:
    output_placeholder.image(
        st.session_state["annotated_image"],
        caption="Detected vehicles",
        width="stretch",
    )
    info_placeholder.markdown(st.session_state["detection_summary"])
    result_bytes = cv2.imencode(".png", cv2.cvtColor(st.session_state["annotated_image"], cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button(
        labels["download"], result_bytes, "vehicle-detection-result.png", "image/png",
        use_container_width=True,
    )

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
            run_id = entry.setdefault("id", f"run-{index}-{entry.get('timestamp', datetime.now()).timestamp()}")
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            timestamp_text = timestamp.strftime("%d %b %Y, %I:%M %p") if timestamp else "Unknown time"
            st.markdown(
                f"**Run {index}** | {timestamp_text} | Source: `{entry.get('source', 'Image')}` | "
                f"Confidence: `{entry['confidence']:.2f}` | IoU: `{entry['iou']:.2f}`"
            )
            action_load, action_delete = st.columns(2)
            with action_load:
                if st.button("📂 Load Run", key=f"load-{run_id}", use_container_width=True):
                    st.session_state["pending_load"] = {
                        "id": run_id,
                        "input": entry["input"],
                        "output": entry["output"],
                        "summary": entry["summary"],
                        "confidence": entry["confidence"],
                        "iou": entry["iou"],
                        "image_size": entry.get("image_size", 640),
                        "max_detections": entry.get("max_detections", 300),
                        "selected_classes": entry.get("selected_classes", list(CLASS_INFO)),
                        "use_tta": entry.get("use_tta", False),
                        "device": entry.get("device", "Auto"),
                        "source": entry.get("source", "Image"),
                    }
                    st.rerun()
            with action_delete:
                if st.button("🗑️ Delete", key=f"delete-{run_id}", use_container_width=True):
                    st.session_state["inference_history"].pop(index - 1)
                    st.rerun()
            history_input, history_output = st.columns(2)
            with history_input:
                st.image(entry["input"], caption="Input image", use_container_width=True)
            with history_output:
                st.image(entry["output"], caption="Detection result", use_container_width=True)
            st.markdown(entry["summary"])
            if index < len(st.session_state["inference_history"]):
                st.divider()
