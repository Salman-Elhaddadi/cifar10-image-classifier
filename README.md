\# CIFAR-10 Image Classifier



A ResNet-9-style CNN trained from scratch in PyTorch on CIFAR-10, with a Gradio demo. \*\*91.04% test accuracy in 10 epochs.\*\*



!\[demo](assets/demo.png)



\## Demo



```bash

pip install -r requirements.txt

python app.py

```



Drop in an image to get the top-5 predictions and the 32x32 version the model actually sees. Below 60% confidence, the app warns that the image may not be one of the 10 classes.



\## Results



| Version | Setup | Val acc | Test acc |

|---|---|---|---|

| 1 | Small VGG-style CNN (32-64-128 channels), no augmentation | 84.41%\* | not reported |

| 2 | Same model + augmentation, proper 45k/5k train/val split | 82.06% | not reported |

| 3 | ResNet-9-style CNN + OneCycle LR (this repo) | 92.04% | \*\*91.04%\*\* |



\\\*Version 1 picked its best epoch on the test set, so that number is optimistic. Version 2 fixes the leak, which is why it scores lower. The final test accuracy is measured once, on the best checkpoint chosen by validation accuracy.



Single run, seed 42. Expect around ±0.3% variation between seeds.



\## Model



`model.py`: conv-BatchNorm-ReLU blocks with two residual blocks (128 and 512 channels), max-pooling, global max pool, dropout (0.2), and a linear classifier. About 6.57M parameters.



\## Training



\- Data: 45,000 train / 5,000 validation split from the CIFAR-10 training set, 10,000 test images untouched

\- Augmentation: random crop (32, padding 4, reflect) and horizontal flip, with CIFAR-10 mean/std normalization

\- Optimizer: Adam (weight decay 1e-4), OneCycleLR with max LR 0.01, batch size 400, gradient clipping at 0.1

\- 10 epochs on a free Kaggle T4 GPU (training on CPU works but is slow)



```bash

python train.py

```



This saves the best-validation checkpoint to `best.pt` and prints the final test accuracy.



\## Limitations



\- Only 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Anything else is forced into one of them.

\- Input is downscaled to 32x32, so the model works best on clear photos with one centered object.

\- Softmax confidence is not calibrated, and out-of-distribution images can still get high scores.



\## Structure



```

.

├── app.py          # Gradio demo

├── model.py        # CNN architecture

├── train.py        # training + evaluation

├── best.pt         # trained weights

├── requirements.txt

└── assets/         # screenshots

```

