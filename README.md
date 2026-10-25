# CIFAR-10 Image Classifier

A deep-learning image classifier built with **PyTorch** and a **ResNet-9-style CNN**, trained from scratch on the CIFAR-10 dataset. The project includes an interactive **Gradio app** that predicts the class of an uploaded image.

<p align="center">
  <strong>91.04% test accuracy · 10 classes · PyTorch</strong>
</p>

## 🎬 Demo

Try the app locally by following the setup instructions below.

<!-- Replace the URL below with the GitHub video-attachment URL after uploading test_video_cifar10.mp4.
     To get one: edit this README on GitHub, drag the MP4 into the editor, wait for the upload,
     and keep the generated https://github.com/user-attachments/assets/... URL. -->

**Watch the application demo:**

`PASTE_GITHUB_VIDEO_ATTACHMENT_URL_HERE`

## ✨ Features

- Classifies uploaded images into one of the 10 CIFAR-10 categories.
- Displays the **top-5 predicted classes** and their confidence scores.
- Shows the resized **32 × 32** image the model receives.
- Displays a warning when the top prediction confidence is below 60%.
- Includes a training script and a saved model checkpoint.

## 🧠 Model & Results

The classifier uses a ResNet-9-style convolutional neural network implemented in PyTorch.

| Experiment | Model / setup | Validation accuracy | Test accuracy |
|---|---|---:|---:|
| 1 | VGG-style CNN, no augmentation | 84.41%* | — |
| 2 | VGG-style CNN with augmentation | 82.06% | — |
| 3 | ResNet-9-style CNN with OneCycleLR | 92.04% | **91.04%** |

\*The first experiment selected its best epoch using the test set, so its validation-style result is optimistic. The final model uses a validation split to select the checkpoint and evaluates on the held-out test set.

**Training setup**

- Dataset split: 45,000 training images, 5,000 validation images, and 10,000 test images.
- Augmentation: random crop and horizontal flip.
- Optimizer: Adam with weight decay.
- Learning-rate schedule: OneCycleLR.
- Batch size: 400.
- Training: 10 epochs; a Kaggle T4 GPU was used for the reported run.

Results are from a single run (seed 42), so small variations may occur between runs.

## 🗂️ Project Structure

```text
cifar10-image-classifier/
├── app.py              # Gradio interface and prediction logic
├── model.py            # CNN architecture
├── train.py            # Training and evaluation
├── best.pt             # Trained model weights
├── requirements.txt    # Python dependencies
└── assets/              # Screenshots and project media
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Salman-Elhaddadi/cifar10-image-classifier.git
cd cifar10-image-classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open the local URL printed by Gradio in your terminal, then upload an image to get predictions.

## 🏋️ Train the Model

To train the model again:

```bash
python train.py
```

The script saves the checkpoint selected using validation accuracy to `best.pt` and reports the final test accuracy. Training on CPU is possible, but may be considerably slower.

## 📦 Dataset

This project uses **CIFAR-10**, a dataset of 60,000 32 × 32 RGB images across 10 classes:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

- 50,000 images are provided for training.
- 10,000 images are provided for testing.

The model resizes uploaded images to 32 × 32 pixels, so clear images with a prominent object generally work best.

## ⚠️ Limitations

- The model only predicts one of the 10 CIFAR-10 classes. An unrelated image may still be assigned to a class.
- Images are reduced to 32 × 32 pixels, which can remove fine details.
- Confidence scores are not calibrated probabilities; a high score does not guarantee a correct prediction.

## 🛠️ Built With

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Gradio](https://www.gradio.app/)

## 👤 Author

**Salman Elhaddadi**

- GitHub: [@Salman-Elhaddadi](https://github.com/Salman-Elhaddadi)

---

If you find this project interesting, consider giving the repository a ⭐.
