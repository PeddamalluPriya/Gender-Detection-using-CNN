# Image-Based Gender Classification using CNN

A deep learning project that classifies face images into two classes using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and OpenCV.

## Project Overview

**Goal:** Build an image classification pipeline that:

1. Loads face images from class-specific folders
2. Preprocesses images using OpenCV
3. Resizes images to `64 × 64`
4. Normalizes pixel values to the `[0, 1]` range
5. Splits the data into training and test sets
6. Trains a CNN using TensorFlow/Keras
7. Evaluates the trained model using test loss and accuracy

## Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Model Architecture

The CNN uses:

- `Conv2D(32, 3×3)` + ReLU
- `MaxPooling2D(2×2)`
- `Conv2D(64, 3×3)` + ReLU
- `MaxPooling2D(2×2)`
- `Flatten`
- `Dense(64)` + ReLU
- `Dense(2)` + Softmax

The model is compiled with the Adam optimizer and categorical cross-entropy loss.

## Data Preprocessing

Images are:

- Loaded with OpenCV
- Resized to `64 × 64`
- Normalized by dividing pixel values by `255.0`
- Converted into NumPy arrays
- Split into training and test sets using an 80/20 split

## Repository Structure

```text
gender-detection-cnn/
├── GenderDetection_CNN.ipynb
├── README.md
└── dataset/
    ├── Female/
    └── Male/
```

> The dataset folders are intentionally not included in this repository. Add your dataset locally under `dataset/Female/` and `dataset/Male/` before running the notebook.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd gender-detection-cnn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the images in:

```text
dataset/Female/
dataset/Male/
```

### 5. Run the notebook

Open `GenderDetection_CNN.ipynb` in Jupyter Notebook or VS Code and run the cells.

## Evaluation

The notebook evaluates the trained CNN using:

- Test accuracy
- Test loss
- Training accuracy
- Validation accuracy

The notebook also plots training and validation accuracy across epochs.

## Key Learning Outcomes

- Practical CNN architecture design
- Image preprocessing with OpenCV
- TensorFlow/Keras model training
- Train/test data preparation
- Validation-based model evaluation
- Understanding image classification workflows

## Future Improvements

- Add data augmentation to improve robustness
- Experiment with dropout and batch normalization
- Tune learning rate, batch size, and network depth
- Compare against transfer-learning architectures
- Add precision, recall, F1-score, and a confusion matrix

## Author

**SmartInternz**
