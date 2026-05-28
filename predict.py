"""
predict.py — Run inference on new images using the trained model.

Usage:
    python predict.py --image path/to/your/image.jpg
    python predict.py --image path/to/image.jpg --top 5   # show top-5 predictions
    python predict.py --folder path/to/folder/            # predict all images in a folder

Requirements:
    - saved_model/gesture_model.keras  (produced by train.py)
    - class_names.json                 (produced by train.py)
"""

import os
import sys
import argparse
import json

import numpy as np
import tensorflow as tf

from dataset_loader import IMAGE_SIZE, CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model + class names
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_classes(
    model_path: str = "saved_model/gesture_model.keras",
    classes_path: str = "class_names.json"
):
    """
    Load the trained Keras model and the associated class name list.

    Args:
        model_path   : Path to the saved .keras model file
        classes_path : Path to class_names.json

    Returns:
        model        : Loaded Keras model
        class_names  : List of gesture class name strings
    """
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        print("  → Did you run train.py first?")
        sys.exit(1)

    if not os.path.exists(classes_path):
        print(f"[Error] Class names file not found: {classes_path}")
        print("  → Did you run train.py first?")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        class_names = json.load(f)

    print(f"[Predict] Model loaded from   : {model_path}")
    print(f"[Predict] {len(class_names)} classes: {class_names}\n")

    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Helper: preprocess a single image
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load one image file and prepare it for model inference.
    Applies the same resizing + normalization used during training.

    Args:
        image_path : Path to a JPG or PNG image

    Returns:
        Float32 NumPy array of shape (1, H, W, C) — batch dimension included
    """
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=CHANNELS, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)   # add batch dim: (1, 64, 64, 3)
    return image.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Predict one image
# ─────────────────────────────────────────────────────────────────────────────

def predict_one(model, class_names: list, image_path: str, top_k: int = 3) -> dict:
    """
    Predict the gesture class for one image.

    Args:
        model       : Loaded Keras model
        class_names : List of class name strings
        image_path  : Path to the image file
        top_k       : How many top predictions to return

    Returns:
        Dict with 'top1_class', 'top1_confidence', and 'top_predictions' list
    """
    img   = preprocess_image(image_path)
    probs = model.predict(img, verbose=0)[0]   # shape (num_classes,)

    # Sort predictions by probability (highest first)
    sorted_indices = np.argsort(probs)[::-1]

    top1_idx  = sorted_indices[0]
    top1_name = class_names[top1_idx]
    top1_conf = probs[top1_idx]

    top_preds = [
        {"class": class_names[i], "confidence": float(probs[i])}
        for i in sorted_indices[:top_k]
    ]

    return {
        "image"          : os.path.basename(image_path),
        "top1_class"     : top1_name,
        "top1_confidence": float(top1_conf),
        "top_predictions": top_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Predict hand gestures with trained model")
    parser.add_argument("--image",  type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    parser.add_argument("--top",    type=int, default=3,    help="Number of top predictions to show")
    parser.add_argument("--model",  type=str, default="saved_model/gesture_model.keras")
    parser.add_argument("--classes", type=str, default="class_names.json")
    return parser.parse_args()


def main():
    args   = parse_args()
    model, class_names = load_model_and_classes(args.model, args.classes)

    VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

    if args.image:
        # ── Single image mode ────────────────────────────────────────────────
        result = predict_one(model, class_names, args.image, top_k=args.top)

        print(f"Image      : {result['image']}")
        print(f"Prediction : {result['top1_class']}  ({result['top1_confidence']*100:.1f}% confidence)")
        print(f"\nTop-{args.top} predictions:")
        for i, p in enumerate(result["top_predictions"], 1):
            bar = "█" * int(p["confidence"] * 30)
            print(f"  {i}. {p['class']:<12} {p['confidence']*100:5.1f}%  {bar}")

    elif args.folder:
        # ── Folder mode: predict all images ──────────────────────────────────
        image_files = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if f.lower().endswith(VALID_EXT)
        ]

        if not image_files:
            print(f"[Error] No images found in: {args.folder}")
            sys.exit(1)

        print(f"Found {len(image_files)} images. Running predictions...\n")
        correct = 0

        for img_path in image_files:
            result = predict_one(model, class_names, img_path, top_k=1)
            print(f"  {result['image']:<30} → {result['top1_class']}  ({result['top1_confidence']*100:.1f}%)")

    else:
        print("Usage examples:")
        print("  python predict.py --image path/to/gesture.jpg")
        print("  python predict.py --folder path/to/images/ --top 5")


if __name__ == "__main__":
    main()
