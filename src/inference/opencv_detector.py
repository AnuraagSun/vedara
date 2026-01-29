import cv2
import numpy as np
import time
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.inference.detector import (
    DetectorInterface,
    Detection,
    BoundingBox,
    InferenceResult,
    get_class_name,
    COCO_CLASSES
)
from src.utils.platform_detect import get_project_root
from src.utils.config_loader import get_config


MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class OpenCVDetector(DetectorInterface):


    MODEL_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
    CONFIG_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"

    def __init__(self) -> None:
        super().__init__()
        self._net: Optional[cv2.dnn.Net] = None
        self._frame_count: int = 0


        self._models_dir = get_project_root() / "models"
        self._model_path = self._models_dir / "mobilenet_ssd.caffemodel"
        self._config_path = self._models_dir / "mobilenet_ssd.prototxt"


        config = get_config()
        self._confidence_threshold = config.inference.confidence_threshold
        self._input_size = (300, 300)

    def _download_model(self) -> bool:

        self._models_dir.mkdir(parents=True, exist_ok=True)

        try:

            if not self._config_path.exists():
                print("[VEDARA] Downloading MobileNet-SSD config...")
                urllib.request.urlretrieve(self.CONFIG_URL, self._config_path)
                print(f"[VEDARA] Saved: {self._config_path}")


            if not self._model_path.exists():
                print("[VEDARA] Downloading MobileNet-SSD model (23MB)...")
                print("[VEDARA] This only happens once...")

                def progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 // total_size)
                    print(f"\r[VEDARA] Progress: {percent}%", end="", flush=True)

                urllib.request.urlretrieve(self.MODEL_URL, self._model_path, progress)
                print("\n[VEDARA] Model downloaded successfully!")

            return True

        except Exception as e:
            print(f"[VEDARA] Download failed: {e}")
            return False

    def initialize(self, model_path: Optional[Path] = None) -> bool:

        print("[VEDARA] Initializing OpenCV DNN Detector...")


        if not self._model_path.exists() or not self._config_path.exists():
            print("[VEDARA] Model not found. Downloading...")
            if not self._download_model():
                self._last_error = "Failed to download model"
                return False

        try:

            self._net = cv2.dnn.readNetFromCaffe(
                str(self._config_path),
                str(self._model_path)
            )


            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self._model_name = "MobileNet-SSD"
            self._input_width = 300
            self._input_height = 300
            self._is_initialized = True

            print(f"[VEDARA] Model loaded: {self._model_name}")
            print(f"[VEDARA] Input size: {self._input_width}x{self._input_height}")
            print(f"[VEDARA] Classes: {len(MOBILENET_CLASSES)}")
            print("[VEDARA] REAL object detection ready!")

            return True

        except Exception as e:
            self._last_error = f"Failed to load model: {e}"
            print(f"[VEDARA] Error: {self._last_error}")
            return False

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = 0.45
    ) -> InferenceResult:

        if not self._is_initialized or self._net is None:
            return InferenceResult(model_name="uninitialized")

        if confidence_threshold is None:
            confidence_threshold = self._confidence_threshold

        self._frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        detections: List[Detection] = []

        start_time = time.perf_counter()

        try:

            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=0.007843,
                size=self._input_size,
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
                crop=False
            )

            preprocess_time = (time.perf_counter() - start_time) * 1000


            inference_start = time.perf_counter()
            self._net.setInput(blob)
            output = self._net.forward()
            inference_time = (time.perf_counter() - inference_start) * 1000


            postprocess_start = time.perf_counter()


            for i in range(output.shape[2]):
                detection = output[0, 0, i]
                confidence = float(detection[2])

                if confidence < confidence_threshold:
                    continue

                class_id = int(detection[1])


                if class_id == 0:
                    continue


                x1 = detection[3]
                y1 = detection[4]
                x2 = detection[5]
                y2 = detection[6]


                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))


                bbox = BoundingBox(
                    x_norm=x1,
                    y_norm=y1,
                    w_norm=x2 - x1,
                    h_norm=y2 - y1
                ).to_pixel_coords(frame_width, frame_height)


                if class_id < len(MOBILENET_CLASSES):
                    class_name = MOBILENET_CLASSES[class_id]
                else:
                    class_name = f"class_{class_id}"

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    track_id=None
                ))

            postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        except Exception as e:
            print(f"[VEDARA] Detection error: {e}")
            preprocess_time = 0
            inference_time = 0
            postprocess_time = 0

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            frame_number=self._frame_count,
            timestamp=time.time(),
            model_name=self._model_name
        )

    def release(self) -> None:

        self._net = None
        self._is_initialized = False

    def get_diagnostics(self) -> Dict[str, Any]:

        base = super().get_diagnostics()
        base.update({
            "detector_type": "OpenCV_DNN",
            "model_file": str(self._model_path),
            "model_exists": self._model_path.exists(),
            "config_exists": self._config_path.exists(),
            "classes": len(MOBILENET_CLASSES),
            "frame_count": self._frame_count,
        })
        return base
