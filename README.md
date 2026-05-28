# Real-Time Hand Gesture Recognition using CNN

A deep learning based **Hand Gesture Recognition System** built using **Python, TensorFlow/Keras, and OpenCV**.
The project uses a custom Convolutional Neural Network trained from scratch to recognize hand gestures and alphabet signs in real time.

The model supports **32 gesture classes**, including alphabets **A-Z** and common gestures such as **hello, bye, yes, no, perfect, and thankyou**.

---

## Features

* Custom CNN model built from scratch
* Real-time hand gesture recognition using webcam
* Supports 32 gesture classes
* Image-based gesture prediction
* Folder-based batch prediction
* Model evaluation on validation dataset
* Per-class accuracy calculation
* Classification report generation
* Confusion matrix generation
* Training history and summary logging
* Camera testing utility
* Dataset capture utility

---

## Gesture Classes

The model recognizes the following classes:

```text
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
Q, R, S, T, U, V, W, X, Y, Z,
bye, hello, no, perfect, thankyou, yes
```

---

## Tech Stack

| Category                | Technology         |
| ----------------------- | ------------------ |
| Programming Language    | Python             |
| Deep Learning Framework | TensorFlow / Keras |
| Computer Vision         | OpenCV             |
| Numerical Computing     | NumPy              |
| Visualization           | Matplotlib         |
| Evaluation Metrics      | Scikit-learn       |
| Image Processing        | Pillow             |

---

## Project Structure

```text
Hand-Gesture-Recognition-CNN/
 ├── capture_faces.py          # Captures gesture/face images using webcam
 ├── class_names.json          # List of gesture classes
 ├── dataset_loader.py         # Loads and preprocesses dataset
 ├── evaluate.py               # Evaluates trained model
 ├── model.py                  # Custom CNN architecture
 ├── predict.py                # Predicts gesture from image or folder
 ├── requirements.txt          # Project dependencies
 ├── test_cameras.py           # Checks available webcam indexes
 ├── train.py                  # Trains the CNN model
 ├── webcam_predict.py         # Real-time webcam prediction
 ├── training_history.png      # Training accuracy/loss graph
 ├── training_log.csv          # Epoch-wise training log
 ├── training_summary.json     # Final training summary
 ├── README.md
 └── .gitignore
```

---

## Model Architecture

The project uses a custom CNN trained from scratch.

Architecture overview:

* Input image size: `64 x 64 x 3`
* 4 Convolutional blocks
* Batch Normalization
* ReLU activation
* Max Pooling
* Global Average Pooling
* Dense layers
* Dropout regularization
* Softmax output layer

The model is designed for multi-class image classification.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Hand-Gesture-Recognition-CNN.git
cd Hand-Gesture-Recognition-CNN
```

### 2. Create a Virtual Environment

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

The dataset should be arranged in the following format:

```text
dataset/
 ├── train/
 │   ├── A/
 │   ├── B/
 │   ├── C/
 │   └── ...
 └── val/
     ├── A/
     ├── B/
     ├── C/
     └── ...
```

Each class folder should contain gesture images belonging to that class.

---

## Training the Model

Run the training script:

```bash
python train.py
```

Or provide a custom dataset path:

```bash
python train.py --data path/to/dataset --epochs 50 --batch 32
```

The training script will:

* Load the dataset
* Detect gesture classes
* Build the CNN model
* Train the model
* Save the best model
* Generate training logs
* Save training summary
* Plot training accuracy and loss curves

---

## Predict Gesture from an Image

```bash
python predict.py --image path/to/image.jpg
```

To show top 5 predictions:

```bash
python predict.py --image path/to/image.jpg --top 5
```

---

## Predict Gestures from a Folder

```bash
python predict.py --folder path/to/images/
```

---

## Real-Time Webcam Prediction

Run:

```bash
python webcam_predict.py
```

Controls:

```text
Q = Quit
S = Save screenshot
P = Pause / Resume
```

---

## Evaluate the Model

```bash
python evaluate.py --data path/to/dataset
```

Evaluation includes:

* Overall accuracy
* Per-class accuracy
* Classification report
* Confusion matrix

---

## Camera Testing

If the webcam does not open, test available camera indexes using:

```bash
python test_cameras.py
```

Then update the `CAMERA_INDEX` value in `webcam_predict.py`.

---

## Training Results

Current training summary:

```text
Number of classes: 32
Total epochs: 20
Best validation accuracy: 99.58%
Final validation loss: 0.0655
```

---

## Files Not Included in GitHub

The following files/folders are intentionally ignored:

```text
dataset/
saved_model/
venv/
.venv/
__pycache__/
*.h5
*.keras
```

These files are either large, machine-specific, or automatically generated.

---

## Future Enhancements

* Add MediaPipe hand landmark detection
* Add support for Indian Sign Language
* Add sentence formation from continuous gestures
* Add text-to-speech output
* Add Flask or Streamlit web interface
* Convert model to TensorFlow Lite
* Deploy as a mobile or web application
* Improve real-world accuracy with more diverse dataset images

---

## Author

Developed as a deep learning and computer vision project using Python, TensorFlow, Keras, and OpenCV.
