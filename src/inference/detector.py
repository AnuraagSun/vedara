import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.utils.platform_detect import get_platform, PLATFORM_RPI
from src.utils.config_loader import get_config


@dataclass
class BoundingBox:

    x_norm: float = 0.0
    y_norm: float = 0.0
    w_norm: float = 0.0
    h_norm: float = 0.0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def to_pixel_coords(self, frame_width: int, frame_height: int) -> 'BoundingBox':

        self.x = int(self.x_norm * frame_width)
        self.y = int(self.y_norm * frame_height)
        self.w = int(self.w_norm * frame_width)
        self.h = int(self.h_norm * frame_height)
        return self

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x, self.y), (self.x + self.w, self.y + self.h))

    @property
    def area(self) -> int:
        return self.w * self.h

    def __repr__(self) -> str:
        return f"BBox({self.x}, {self.y}, {self.w}x{self.h})"


@dataclass
class Detection:

    class_id: int = 0
    class_name: str = "unknown"
    confidence: float = 0.0
    bbox: BoundingBox = field(default_factory=BoundingBox)
    track_id: Optional[int] = None

    def __repr__(self) -> str:
        return f"Detection({self.class_name}: {self.confidence:.2f} @ {self.bbox})"


@dataclass
class InferenceResult:

    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0
    model_name: str = "unknown"

    @property
    def total_time_ms(self) -> float:
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

    @property
    def detection_count(self) -> int:
        return len(self.detections)


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def get_class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


class DetectorInterface(ABC):


    def __init__(self) -> None:
        self._is_initialized: bool = False
        self._model_name: str = "unknown"
        self._input_width: int = 300
        self._input_height: int = 300
        self._last_error: Optional[str] = None

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self._input_width, self._input_height)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @abstractmethod
    def initialize(self, model_path: Optional[Path] = None) -> bool:
        pass

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45
    ) -> InferenceResult:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "initialized": self._is_initialized,
            "model_name": self._model_name,
            "input_size": f"{self._input_width}x{self._input_height}",
            "last_error": self._last_error
        }


try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
        TFLITE_SOURCE = "tensorflow"
    except ImportError:
        TFLITE_AVAILABLE = False
        TFLITE_SOURCE = None
        tflite = None


class TFLiteDetector(DetectorInterface):


    def __init__(self) -> None:
        super().__init__()
        self._interpreter = None
        self._frame_count = 0

        config = get_config()
        self._num_threads = config.inference.threads
        self._confidence_threshold = config.inference.confidence_threshold

    def initialize(self, model_path: Optional[Path] = None) -> bool:
        if not TFLITE_AVAILABLE:
            self._last_error = "TFLite not available"
            return False

        if model_path is None:
            config = get_config()
            model_path = config.inference.model_path

        if not model_path.exists():
            self._last_error = f"Model not found: {model_path}"
            return False

        try:
            self._interpreter = tflite.Interpreter(
                model_path=str(model_path),
                num_threads=self._num_threads
            )
            self._interpreter.allocate_tensors()

            input_details = self._interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            self._input_height = input_shape[1]
            self._input_width = input_shape[2]

            self._model_name = model_path.stem
            self._is_initialized = True

            print(f"[VEDARA] TFLite model loaded: {self._model_name}")
            return True

        except Exception as e:
            self._last_error = f"TFLite init error: {e}"
            return False

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45
    ) -> InferenceResult:

        return InferenceResult(model_name=self._model_name)

    def release(self) -> None:
        self._interpreter = None
        self._is_initialized = False


class MockDetector(DetectorInterface):


    def __init__(self) -> None:
        super().__init__()
        self._frame_count = 0

    def initialize(self, model_path: Optional[Path] = None) -> bool:
        self._model_name = "MOCK_DETECTOR"
        self._is_initialized = True
        print("[VEDARA] WARNING: Using MOCK detector - detections are FAKE!")
        return True

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45
    ) -> InferenceResult:
        self._frame_count += 1
        frame_height, frame_width = frame.shape[:2]


        t = (self._frame_count % 100) / 100.0
        x = 0.1 + 0.5 * abs(np.sin(t * np.pi * 2))

        bbox = BoundingBox(
            x_norm=x, y_norm=0.2, w_norm=0.15, h_norm=0.4
        ).to_pixel_coords(frame_width, frame_height)

        return InferenceResult(
            detections=[
                Detection(
                    class_id=0,
                    class_name="FAKE_person",
                    confidence=0.95,
                    bbox=bbox
                )
            ],
            inference_time_ms=15.0,
            frame_number=self._frame_count,
            model_name=self._model_name
        )

    def release(self) -> None:
        self._is_initialized = False


class DetectorFactory:


    @staticmethod
    def create(
        model_path: Optional[Path] = None,
        use_mock_if_unavailable: bool = False,
        force_opencv: bool = True
    ) -> DetectorInterface:


        if force_opencv:
            try:
                from src.inference.opencv_detector import OpenCVDetector
                detector = OpenCVDetector()
                if detector.initialize():
                    print("[VEDARA] Using OpenCV DNN for REAL object detection!")
                    return detector
            except ImportError as e:
                print(f"[VEDARA] OpenCV detector not available: {e}")
            except Exception as e:
                print(f"[VEDARA] OpenCV detector init failed: {e}")


        if TFLITE_AVAILABLE:
            config = get_config()
            tflite_path = model_path or config.inference.model_path

            if tflite_path.exists():
                detector = TFLiteDetector()
                if detector.initialize(tflite_path):
                    print("[VEDARA] Using TFLite for object detection!")
                    return detector


        if use_mock_if_unavailable:
            print("[VEDARA] WARNING: Falling back to MOCK detector!")
            print("[VEDARA] Detections will be FAKE and not from camera!")
            detector = MockDetector()
            detector.initialize()
            return detector

        raise RuntimeError("No detector available. Install OpenCV or provide TFLite model.")

    @staticmethod
    def get_available_backends() -> Dict[str, bool]:

        opencv_available = False
        try:
            import cv2
            opencv_available = hasattr(cv2, 'dnn')
        except ImportError:
            pass

        return {
            "opencv_dnn": opencv_available,
            "tflite": TFLITE_AVAILABLE,
            "mock": True
        }


def create_detector(
    model_path: Optional[Path] = None,
    use_mock_if_unavailable: bool = False
) -> DetectorInterface:

    return DetectorFactory.create(model_path, use_mock_if_unavailable, force_opencv=True)


if __name__ == "__main__":
    import cv2

    print("=" * 60)
    print("VEDARA Detector Test - REAL Object Detection")
    print("=" * 60)


    print("\nAvailable backends:")
    for name, available in DetectorFactory.get_available_backends().items():
        print(f"  {name}: {'YES' if available else 'NO'}")


    print("\nCreating REAL detector...")
    detector = create_detector(use_mock_if_unavailable=True)
    print(f"Detector: {type(detector).__name__}")
    print(f"Model: {detector.model_name}")


    print("\nTesting detection...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect(test_frame)

    print(f"Detections: {result.detection_count}")
    print(f"Inference time: {result.inference_time_ms:.1f}ms")

    for det in result.detections:
        print(f"  - {det.class_name}: {det.confidence:.2f}")

    detector.release()
    print("\n" + "=" * 60)
