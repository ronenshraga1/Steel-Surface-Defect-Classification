# Steel Surface Defect Classification

## Project overview

I built and compared deep learning models to classify six types of steel surface defects from images. I started with a small CNN, tried data augmentation and a deeper CNN, then used MobileNetV2 transfer learning. The final model reached **98.89% accuracy on the held-out test set**. The [notebook](Steel_Surface_Defect_Classification.ipynb) contains the experiments, evaluation plots, and Grad-CAM examples.

## Dataset and preprocessing

I used the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database). Its six balanced classes are inclusion, scratches, pitted surface, patches, crazing, and rolled-in scale.

| Split | Images |
|---|---:|
| Training | 1,440 |
| Validation | 180 |
| Test | 180 |
| **Total** | **1,800** |

The source images are grayscale. I decode them as RGB for the models, resize them to 200 × 200, convert them to `float32`, and scale pixel values to [0, 1]. A TensorFlow `tf.data` pipeline handles loading, batching, training-data shuffling, and prefetching. I also checked SHA-256 hashes across the splits and found no exact duplicate files; that check does not detect near-duplicates.

The original validation/test split shuffled filenames returned by `os.listdir()` with seed 42. Because that file order was not saved, a new run may assign different images to the two sets and may not reproduce the recorded scores exactly.

## Model development

### Baseline CNN

I first trained a CNN with two convolution and max-pooling blocks, followed by global average pooling, a 64-unit dense layer, and a six-class softmax output. I trained for 10 epochs, loaded the checkpoint with the lowest validation loss, and continued to epoch 20. This gave me a baseline of **76.67% validation accuracy**.

### Data augmentation

I kept the same CNN and added horizontal and vertical flips, small rotations (`RandomRotation(0.05)`), and contrast adjustments. I wanted the model to rely less on the exact appearance of each training image. Validation accuracy was **75.00%**: augmentation improved validation loss in this run, but not accuracy.

### Lower learning rate

I continued the augmented model with Adam's learning rate reduced from 0.001 to 0.0001 and used early stopping. Validation accuracy was **74.44%**, so this change did not improve the result.

### Deeper CNN

I added a third convolution and pooling block to see whether the model could learn more complex texture patterns. With augmentation, it reached **83.33% validation accuracy**, the best result from the CNNs I trained from scratch.

### MobileNetV2 transfer learning

For the final experiment, I used MobileNetV2 pretrained on ImageNet without its original classification head. I froze the backbone, kept its BatchNormalization layers in inference mode, and trained a new global-average-pooling and six-class softmax head. Pretrained features were useful on this relatively small dataset: the model reached **97.78% validation accuracy**. I did not fine-tune the backbone.

The saved model includes a Rescaling layer that converts the pipeline's [0, 1] inputs to [-1, 1] for MobileNetV2. The [class mapping](class_names.json) records the exact order of its six outputs.

## Model comparison

These are the recorded validation results. Checkpoints were selected using validation loss.

| Model | Validation accuracy |
|---|---:|
| Baseline CNN | 76.67% |
| CNN + data augmentation | 75.00% |
| CNN + augmentation + lower learning rate | 74.44% |
| Deeper CNN + augmentation | 83.33% |
| MobileNetV2 transfer learning | 97.78% |

The deeper CNN improved on the baseline, but the frozen MobileNetV2 features worked substantially better than the CNNs trained from scratch.

## Final test evaluation

I selected the MobileNetV2 checkpoint using validation results, then evaluated it on the held-out test set. It classified **178 of 180 images correctly**, for **98.89% test accuracy** and **0.0425 test loss**. Both misclassified images belonged to the pitted surface class. The notebook includes the test classification report, confusion matrix, and error examples.

## Grad-CAM

I used Grad-CAM on MobileNetV2's `out_relu` feature maps to inspect one correct and one incorrect test prediction. The notebook shows each image with its heatmap or overlay. Because the feature maps are only 7 × 7 before resizing, the highlights are coarse; they show which regions affected a class score, not a precise outline of a physical defect.

## Streamlit demo

The [Streamlit app](app.py) provides a small demo of the saved model. It accepts a JPG or PNG image and shows the predicted class, scores for all six classes, and a Grad-CAM overlay. It needs [the trained model](steel_defect_transfer.keras) and `class_names.json` in the project directory.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Technologies

Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, KaggleHub, Google Colab, and Streamlit. The notebook uses Colab and Google Drive for its experiments; `requirements.txt` covers the local app.

## Repository files

```text
Steel-Surface-Defect-Classification/
├── Steel_Surface_Defect_Classification.ipynb  # experiments and results
├── steel_defect_transfer.keras                 # final saved model
├── class_names.json                            # model output order
├── app.py                                      # Streamlit demo
├── requirements.txt                            # app dependencies
└── README.md
```
