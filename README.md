# Steel Surface Defect Classification

A Streamlit app for classifying six steel surface defects with a MobileNetV2 model trained on the NEU Surface Defect Database. The accompanying notebook documents the CNN experiments, transfer learning, and final evaluation.

Upload a JPG or PNG image to see the predicted class, all six prediction scores, and a Grad-CAM overlay. The class order comes from `class_names.json` and matches the saved model's output.

The saved MobileNetV2 model achieved **98.89% accuracy (178/180 images)** on the project's held-out test set. This is a result from that evaluation, not a guarantee for new images.

## Run locally

Use Python 3.10–3.13, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `steel_defect_transfer.keras` and `class_names.json` next to `app.py`. The app accepts RGB images at any size, resizes them to 200 × 200 with the notebook's bilinear preprocessing, and scales pixel values to [0, 1]. The saved model performs the additional scaling needed by MobileNetV2.

Grad-CAM uses the saved model's `out_relu` feature maps to show which image regions affected the predicted class score. Its heatmap is coarse and does not outline or verify a physical defect.
