import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = "yolo_weights/ppe.pt"

st.set_page_config(page_title="PPE Detection", layout="centered")
st.title("PPE Detection")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
name_to_id = {name: idx for idx, name in model.names.items()}

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
selected_classes = st.sidebar.multiselect(
    "Classes to detect",
    options=list(name_to_id.keys()),
    default=list(name_to_id.keys()),
)
class_ids = [name_to_id[name] for name in selected_classes]

file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if file is not None and not class_ids:
    st.warning("Select at least one class in the sidebar.")
elif file is not None:
    is_video = file.type.startswith("video")

    if not is_video:
        results = model(file.getvalue(), conf=confidence, classes=class_ids)
        annotated = results[0].plot()
        st.image(annotated[:, :, ::-1], caption="Detections", use_container_width=True)

        ok, encoded = cv2.imencode(".png", annotated)
        if ok:
            st.download_button("Download image", encoded.tobytes(), file_name="detected.png", mime="image/png")
    else:
        in_file = tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split(".")[-1])
        in_file.write(file.read())

        cap = cv2.VideoCapture(in_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_area = st.empty()
        progress = st.progress(0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model(frame, conf=confidence, classes=class_ids)
            annotated = results[0].plot()
            frame_area.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
            writer.write(annotated)

            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()

        with open(out_path, "rb") as f:
            st.download_button("Download video", f.read(), file_name="detected.mp4", mime="video/mp4")
