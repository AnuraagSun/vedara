from src.inference.detector import (

    BoundingBox,
    Detection,
    InferenceResult,


    COCO_CLASSES,
    get_class_name,


    DetectorInterface,


    TFLiteDetector,
    MockDetector,


    DetectorFactory,
    create_detector,


    TFLITE_AVAILABLE,
    TFLITE_SOURCE,
)


try:
    from src.inference.opencv_detector import OpenCVDetector, MOBILENET_CLASSES
except ImportError:
    OpenCVDetector = None
    MOBILENET_CLASSES = []

__all__ = [
    "BoundingBox",
    "Detection",
    "InferenceResult",
    "COCO_CLASSES",
    "MOBILENET_CLASSES",
    "get_class_name",
    "DetectorInterface",
    "TFLiteDetector",
    "OpenCVDetector",
    "MockDetector",
    "DetectorFactory",
    "create_detector",
    "TFLITE_AVAILABLE",
    "TFLITE_SOURCE",
]
