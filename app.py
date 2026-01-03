import os
import streamlit as st
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import requests

# =========================
# Config
# =========================
st.set_page_config(page_title="DR Classifier (0-4)", layout="centered")

MODEL_URL = "https://github.com/rickyperdana22-cpu/MLDiabeticRetinopathy_RS/releases/download/V1/best_rs_rehearsal_head.pth"
MODEL_PATH = "best_rs_rehearsal_head.pth"

CLASS_NAMES = {
    0: "0 - No DR",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Proliferative DR",
}

# =========================
# Utils
# =========================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_model()

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=5
    )

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device

infer_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(model, device, pil_img):
    x = infer_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

# =========================
# UI (INI ASLI PUNYA KAMU)
# =========================
st.title("Diabetic Retinopathy Classifier (0–4)")
st.write("Upload gambar fundus → model akan memprediksi tingkat DR (0–4) + confidence.")

# Load model
try:
    model, device = load_model()
    st.success(f"Model loaded ✅ ({'GPU' if device.type == 'cuda' else 'CPU'})")
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload gambar fundus (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    st.subheader("Preview")
    st.image(img, use_container_width=True)

    if st.button("Prediksi"):
        pred, conf, probs = predict(model, device, img)

        st.subheader("Hasil Prediksi")
        st.metric(
            "Predicted Class",
            f"{pred} ({CLASS_NAMES.get(pred)})"
        )
        st.metric("Confidence", f"{conf:.3f}")

        st.subheader("Probabilitas per kelas")
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
        st.bar_chart(prob_dict)

        st.info(
            "Catatan: model ini diadaptasi ke data RS yang jumlahnya terbatas, "
            "sehingga performa per kelas bisa tidak merata."
        )
else:
    st.caption("Tip: pakai gambar RS untuk demo agar hasilnya lebih relevan.")
