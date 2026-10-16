"""
SafeRide AI - YOLOv8 Training Script
Model: YOLOv8n (Nano) for Helmet & Two-Wheeler Detection
Classes:
  0: helmet
  1: without (Helmet violation)
  2: two_wheeler
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train SafeRide AI YOLOv8 model on helmet dataset")
    parser.add_argument("--data", type=str, default=str(Path(__file__).parent.parent / "config" / "dataset.yaml"),
                        help="Path to dataset YAML configuration")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model checkpoint (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8, conservative for 8GB RAM CPU)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on ('cpu', '0', etc.)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Dataloader workers (default: 2 for Windows stability)")
    parser.add_argument("--project", type=str, default=str(Path(__file__).parent.parent / "runs"),
                        help="Directory to save runs")
    parser.add_argument("--name", type=str, default="helmet_yolov8n_run",
                        help="Run experiment name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print(" SafeRide AI - Model Training Setup")
    print("=" * 60)
    print(f"Dataset Config : {args.data}")
    print(f"Base Model     : {args.model}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Batch Size     : {args.batch}")
    print(f"Device         : {args.device}")
    print(f"Output Project : {args.project}")
    print(f"Experiment     : {args.name}")
    print("=" * 60)

    # Validate dataset config path
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset configuration not found at {args.data}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: 'ultralytics' is not installed in the current environment.")
        print("Please activate your virtual environment and run:")
        print("    pip install ultralytics torch torchvision")
        sys.exit(1)

    # Initialize model
    print(f"\n[1/3] Loading base model: {args.model} ...")
    model = YOLO(args.model)

    # Start training
    print(f"[2/3] Starting training on {args.device} ...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Locate and copy best weights to SafeRide-AI/models/best.pt
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    target_weights = models_dir / "best.pt"

    print("\n[3/3] Training finished!")
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"Successfully copied best weights to: {target_weights}")
    else:
        print(f"Warning: Expected weights at {best_weights} not found. Check {args.project}")

    print("\nTraining complete! You can now run validation or inference.")

if __name__ == "__main__":
    main()
