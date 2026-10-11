"""Streamlit demo for the saved steel surface defect classifier."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "steel_defect_transfer.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.json"
IMAGE_SIZE = (200, 200)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file is missing: {MODEL_PATH.name}")
    return keras.models.load_model(MODEL_PATH, compile=False)


def load_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class mapping is missing: {CLASS_NAMES_PATH.name}")
    names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != 6 or any(not isinstance(name, str) for name in names):
        raise ValueError("class_names.json must contain six class names in model output order.")
    if len(set(names)) != len(names):
        raise ValueError("class_names.json contains duplicate class names.")
    return names


def read_image(data):
    with Image.open(BytesIO(data)) as opened_image:
        if opened_image.format not in {"JPEG", "PNG"}:
            raise ValueError("Please upload a JPG or PNG image.")
        opened_image.load()
        return opened_image.convert("RGB")


def preprocess_image(image):
    """Match the notebook: RGB, bilinear 200×200 resize, float32, [0, 1]."""
    rgb_pixels = np.asarray(image.convert("RGB"))
    resized = tf.image.resize(rgb_pixels, IMAGE_SIZE, method="bilinear")
    return tf.expand_dims(tf.cast(resized, tf.float32) / 255.0, axis=0)


def predict(model, image_batch, class_names):
    if model.input_shape[1:] != (200, 200, 3) or model.output_shape[-1] != len(class_names):
        raise ValueError("The saved model shape does not match the image size or class mapping.")
    scores = np.asarray(model(image_batch, training=False)[0], dtype=np.float32)
    if scores.shape != (len(class_names),) or not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction scores.")
    return scores


def make_gradcam_heatmap(model, image_batch, scores, image_size):
    """Follow the saved augmentation → rescaling → backbone → classifier path."""
    augmentation = next(
        (layer for layer in model.layers if isinstance(layer, keras.Sequential)), None
    )
    rescaling = next(
        (layer for layer in model.layers if isinstance(layer, keras.layers.Rescaling)), None
    )
    backbone = next(
        (layer for layer in model.layers
         if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower()), None
    )
    pooling = next(
        (layer for layer in model.layers if isinstance(layer, keras.layers.GlobalAveragePooling2D)), None
    )
    classifier = next(
        (layer for layer in model.layers if isinstance(layer, keras.layers.Dense)), None
    )
    if any(layer is None for layer in (augmentation, rescaling, backbone, pooling, classifier)):
        raise ValueError("The saved model does not have the expected MobileNetV2 layers.")
    if not np.isclose(rescaling.scale, 2.0) or not np.isclose(rescaling.offset, -1.0):
        raise ValueError("The saved model uses a different input scaling layer.")
    if backbone.trainable:
        raise ValueError("The saved MobileNetV2 backbone is not frozen as expected.")

    feature_extractor = keras.Model(backbone.input, backbone.get_layer("out_relu").output)
    augmented = augmentation(image_batch, training=False)
    scaled = rescaling(augmented)
    predicted_index = int(np.argmax(scores))

    with tf.GradientTape() as tape:
        feature_maps = feature_extractor(scaled, training=False)
        tape.watch(feature_maps)
        reconstructed_scores = classifier(pooling(feature_maps))
        class_score = reconstructed_scores[:, predicted_index]

    # A nested backbone can be disconnected from the outer model's graph.
    # Check that this explicit path reproduces the model's actual prediction.
    if not np.allclose(reconstructed_scores.numpy()[0], scores, rtol=1e-4, atol=1e-5):
        raise ValueError("Grad-CAM path does not match the saved model's prediction.")
    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise ValueError("Grad-CAM could not compute gradients for this prediction.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], image_size[::-1], method="bilinear")
    return np.clip(np.asarray(heatmap[..., 0]), 0.0, 1.0)


def make_overlay(image, heatmap, transparency):
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = plt.get_cmap("turbo")(heatmap)[..., :3] * 255.0
    weight = (transparency / 100.0) * heatmap[..., np.newaxis]
    overlay = original * (1.0 - weight) + colored * weight
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))


def show_score_chart(scores, class_names):
    order = np.argsort(scores)[::-1]
    labels = [class_names[index].replace("_", " ").title() for index in order][::-1]
    values = (scores[order] * 100)[::-1]
    colors = ["#356080"] * len(values)
    colors[-1] = "#00b6a0"
    fig, axis = plt.subplots(figsize=(7.5, 3.2))
    axis.barh(labels, values, color=colors, height=0.62)
    axis.set_xlim(0, 108)
    axis.set_xlabel("Prediction score (%)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
    for row, value in enumerate(values):
        axis.text(value + 1, row, f"{value:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def main():
    st.set_page_config(
        page_title="Steel Surface Defect Classification",
        page_icon="🔬",
        layout="wide",
    )
    st.markdown(
        """<style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        h1, h2, h3 {letter-spacing: -0.03em;}
        [data-testid="stMetricValue"] {color: #00a992;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Steel Surface Defect Classification")
    st.caption("Deep Learning Image Classification with MobileNetV2")
    st.write("Upload a steel surface image to see the model's prediction and a Grad-CAM explanation.")

    try:
        class_names = load_class_names()
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file is missing: {MODEL_PATH.name}")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        st.stop()

    st.subheader("Classify an image")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
    image = None
    file_key = None
    if uploaded_file is not None:
        raw_image = uploaded_file.getvalue()
        file_key = hashlib.sha256(raw_image).hexdigest()
        try:
            image = read_image(raw_image)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            st.error(f"Could not read this image: {error}")
        else:
            st.image(image, caption="Uploaded image", width=360)

    if st.button("Classify Image", type="primary", disabled=image is None):
        try:
            with st.spinner("Analyzing image…"):
                model = load_model()
                image_batch = preprocess_image(image)
                scores = predict(model, image_batch, class_names)
                result = {"file_key": file_key, "scores": scores, "heatmap": None, "gradcam_error": None}
                try:
                    result["heatmap"] = make_gradcam_heatmap(model, image_batch, scores, image.size)
                except Exception as error:
                    result["gradcam_error"] = str(error)
                st.session_state["classification"] = result
        except Exception as error:
            st.session_state.pop("classification", None)
            st.error(f"Classification failed: {error}")

    result = st.session_state.get("classification")
    if result is not None and file_key == result["file_key"] and image is not None:
        scores = result["scores"]
        predicted_index = int(np.argmax(scores))
        st.subheader("Prediction")
        label_column, score_column = st.columns(2)
        label_column.metric("Defect class", class_names[predicted_index].replace("_", " ").title())
        score_column.metric("Prediction score", f"{scores[predicted_index]:.1%}")
        st.caption("Scores are softmax outputs, not calibrated confidence estimates.")
        show_score_chart(scores, class_names)

        st.subheader("Grad-CAM")
        if result["gradcam_error"]:
            st.error(f"Grad-CAM could not be generated: {result['gradcam_error']}")
        else:
            transparency = st.slider("Overlay transparency", 0, 100, 55, step=5)
            original_column, overlay_column = st.columns(2)
            original_column.image(image, caption="Original image", use_container_width=True)
            overlay_column.image(
                make_overlay(image, result["heatmap"], transparency),
                caption="Grad-CAM overlay",
                use_container_width=True,
            )
            with st.expander("Show heatmap"):
                heatmap_image = np.uint8(plt.get_cmap("turbo")(result["heatmap"])[..., :3] * 255)
                st.image(heatmap_image, caption="Grad-CAM heatmap", use_container_width=True)
        st.caption(
            "Grad-CAM highlights regions that influenced this class score. It is a coarse visual "
            "explanation, not a precise defect outline or proof that the model found a physical defect."
        )

    st.divider()
    st.subheader("About the model")
    about_columns = st.columns(4)
    for column, label, value in zip(
        about_columns,
        ["Architecture", "Dataset", "Classes", "Input size"],
        ["MobileNetV2", "NEU Surface Defect Database", "6", "200 × 200 RGB"],
    ):
        column.metric(label, value)
    st.write("Held-out test result from the project: **98.89% accuracy (178/180 images)**. Results on new images may differ.")


if __name__ == "__main__":
    main()
