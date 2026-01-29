import urllib.request
import ssl
import sys
import hashlib
from pathlib import Path


def get_project_root() -> Path:

    return Path(__file__).parent.parent.resolve()


def download_file(url: str, dest_path: Path, description: str = "file") -> bool:

    try:
        print(f"Attempting: {url[:60]}...")


        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE


        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Vedara/1.0'}
        )

        with urllib.request.urlopen(request, context=ctx, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    if total_size > 0:
                        percent = downloaded * 100 // total_size
                        mb_down = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        sys.stdout.write(
                            f"\r  Progress: {percent}% ({mb_down:.1f}/{mb_total:.1f} MB)    "
                        )
                        sys.stdout.flush()

            print()
            return True

    except Exception as e:
        print(f"  Failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_model() -> bool:

    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "yolov5n.tflite"

    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"[VEDARA] Model already exists: {model_path}")
        print(f"[VEDARA] Size: {size_mb:.1f} MB")
        return True

    print("=" * 60)
    print("VEDARA Model Download Utility")
    print("=" * 60)


    model_sources = [

        {
            "name": "HuggingFace Mirror",
            "url": "https://huggingface.co/nickmuchi/yolov5n-coco/resolve/main/yolov5n.tflite",
        },

        {
            "name": "GitHub Releases (if available)",
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.tflite",
        },
    ]

    print(f"\nDestination: {model_path}")
    print(f"\nTrying {len(model_sources)} download sources...")
    print("-" * 60)

    for i, source in enumerate(model_sources, 1):
        print(f"\n[{i}/{len(model_sources)}] {source['name']}")

        if download_file(source['url'], model_path, "model"):
            if model_path.exists() and model_path.stat().st_size > 1000000:
                size_mb = model_path.stat().st_size / 1024 / 1024
                print(f"\n[OK] Download successful!")
                print(f"[OK] File size: {size_mb:.1f} MB")
                print(f"[OK] Saved to: {model_path}")
                return True
            else:
                print("  Downloaded file too small, trying next source...")
                if model_path.exists():
                    model_path.unlink()

    print("\n" + "-" * 60)
    print("[!] All automatic download sources failed.")
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
Option 1: Export from Ultralytics (Recommended)
-----------------------------------------------
1. Install ultralytics: pip install ultralytics
2. Run Python:

   from ultralytics import YOLO
   model = YOLO('yolov5n.pt')
   model.export(format='tflite')

3. Copy the exported file to: models/yolov5n.tflite

Option 2: Use ONNX to TFLite conversion
-----------------------------------------------
1. Download ONNX model from:
   https://github.com/ultralytics/yolov5/releases
2. Convert using tf2onnx or similar tool

Option 3: Continue with Mock Detector
-----------------------------------------------
The mock detector works without a model file.
Run: python -m src.inference

""")
    return False


def create_dummy_model() -> bool:

    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    dummy_path = models_dir / "dummy_model.tflite"

    print("\n" + "=" * 60)
    print("Creating Dummy Model for Pipeline Testing")
    print("=" * 60)

    try:
        import numpy as np


        try:
            import tensorflow as tf


            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(320, 320, 3)),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(85)
            ])


            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            with open(dummy_path, 'wb') as f:
                f.write(tflite_model)

            print(f"[OK] Dummy model created: {dummy_path}")
            print(f"[OK] Size: {dummy_path.stat().st_size / 1024:.1f} KB")
            print("\nNote: This is a DUMMY model for testing only.")
            print("      It will NOT produce valid detections.")
            return True

        except ImportError:
            print("[!] TensorFlow not available for dummy model creation.")
            print("    Using mock detector instead.")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to create dummy model: {e}")
        return False


def main():

    print("\n")


    success = download_model()

    if not success:

        print("\nWould you like to create a dummy model for testing? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                success = create_dummy_model()
        except:
            pass


    print("\n" + "=" * 60)
    print("CONFIGURATION REMINDER")
    print("=" * 60)

    project_root = get_project_root()
    models_dir = project_root / "models"

    print(f"\nModels directory: {models_dir}")
    print("Files present:")

    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.suffix == '.tflite':
                print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nTo use a model, update config/config.yaml:")
    print('  inference:')
    print('    model_file: "yolov5n.tflite"  # or your model name')

    print("\n" + "=" * 60)
    if success:
        print("Ready for testing: python -m src.inference")
    else:
        print("Mock detector available: python -m src.inference")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
