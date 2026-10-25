# CIFAR-10 Image Classifier

A deep-learning image classifier built with **PyTorch** and a **ResNet-9-style convolutional neural network**, trained on the CIFAR-10 dataset. An interactive **Gradio** application lets users upload an image and inspect the model's predictions.

<p align="center">
  <strong>91.04% test accuracy · 10 classes · PyTorch</strong>
</p>

## Demo

Watch the application demo below.

<!-- Upload the MP4 through GitHub's README editor and replace the placeholder with the
     generated https://github.com/user-attachments/assets/... URL. -->

PASTE_GITHUB_VIDEO_ATTACHMENT_URL_HERE

## Features

- Predicts one of the 10 CIFAR-10 categories for an uploaded image.
- Displays the top-5 predicted classes and their confidence scores.
- Shows the 32 × 32 image passed to the model.
- Displays a warning when the top prediction confidence is below 60%.
- Includes the model architecture, training script, and saved checkpoint.

## Model and Results

The final classifier is a ResNet-9-style CNN implemented in PyTorch.

| Experiment | Model / setup | Reported accuracy |
|---|---|---:|
| 1 | VGG-style CNN, no augmentation | 84.41%* |
| 2 | VGG-style CNN with augmentation | 82.06% |
| 3 | ResNet-9-style CNN with OneCycleLR | **91.04% test accuracy** |

\*In the first experiment, the best epoch was selected using the test set, so its reported result is not a clean held-out evaluation. The final model selects its checkpoint using validation accuracy and evaluates on the test set.

### Training configuration

- Dataset split: 45,000 training images, 5,000 validation images, and 10,000 test images.
- Augmentation: random crop and horizontal flip.
- Optimizer: Adam with weight decay.
- Learning-rate scheduler: OneCycleLR.
- Batch size: 400.
- Training duration: 10 epochs.
- Reported training hardware: Kaggle T4 GPU.

Results are from one run with seed 42; results may vary between runs.

## Project Structure

```text
cifar10-image-classifier/
├── app.py              # Gradio interface and prediction logic
├── model.py            # CNN architecture
├── train.py            # Training and evaluation
├── best.pt              # Trained model checkpoint
└── requirements.txt    # Python dependencies
```

## Getting Started

### Requirements

- Python 3.10 or later recommended
- pip
- Internet access on the first run if the dataset needs to be downloaded

### 1. Clone the repository

```bash
git clone https://github.com/Salman-Elhaddadi/cifar10-image-classifier.git
cd cifar10-image-classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
python app.py
```

Open the local Gradio URL printed in the terminal, upload an image, and review the predictions.

## Train the Model

To train the model again, run:

```bash
python train.py
```

The training script selects the checkpoint using validation accuracy and reports performance on the test set. Training on a CPU is possible but may take considerably longer.

## Dataset

This project uses **CIFAR-10**, a dataset of 60,000 small, colour RGB images, each sized 32 × 32 pixels, across 10 categories.

| Category | Category | Category |
|---|---|---|
| Airplane | Automobile | Bird |
| Cat | Deer | Dog |
| Frog | Horse | Ship |
| Truck | | |

- 50,000 images are provided for training.
- 10,000 images are provided for testing.

Uploaded images are resized to 32 × 32 pixels before inference. Images with a clear, prominent subject are generally easier to classify.

## Limitations

- The classifier only predicts among the 10 CIFAR-10 categories. Unrelated images will still be assigned to one of these classes.
- Resizing images to 32 × 32 pixels removes fine details.
- Confidence scores should not be interpreted as calibrated probabilities or a guarantee of correctness.

## Technology Stack

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Gradio](https://www.gradio.app/)

## Author

**Salman Elhaddadi**

[GitHub Profile](https://github.com/Salman-Elhaddadi)

---

If you find this project useful, consider giving the repository a star.
