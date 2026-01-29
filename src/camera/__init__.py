from typing import Optional, Type
from pathlib import Path

from src.camera.interface import (
    CameraInterface,
    CameraState,
    CameraProperties,
    FrameData,
    NullCamera
)
from src.camera.opencv_backend import OpenCVCamera, OPENCV_AVAILABLE
from src.camera.picamera_backend import PicameraBackend, PICAMERA2_AVAILABLE
from src.utils.platform_detect import get_platform, PLATFORM_RPI, PLATFORM_WINDOWS
from src.utils.config_loader import get_config


class CameraFactory:


    @staticmethod
    def create(
        backend_override: Optional[str] = None,
        use_null_if_unavailable: bool = True
    ) -> CameraInterface:

        platform = get_platform()
        config = get_config()


        if backend_override is not None:
            return CameraFactory._create_by_name(backend_override)


        backend_string = config.camera.backend


        if backend_string.startswith("picamera2"):
            if PICAMERA2_AVAILABLE:
                print("[VEDARA] Operator, selecting Picamera2 backend for RPi.")
                return PicameraBackend(backend_string)
            elif OPENCV_AVAILABLE:
                print("[VEDARA] Operator, Picamera2 unavailable. Falling back to OpenCV.")
                return OpenCVCamera("opencv:0")

        elif backend_string.startswith("opencv"):
            if OPENCV_AVAILABLE:
                print(f"[VEDARA] Operator, selecting OpenCV backend: {backend_string}")
                return OpenCVCamera(backend_string)


        if platform == PLATFORM_RPI:
            if PICAMERA2_AVAILABLE:
                print("[VEDARA] Operator, RPi detected. Using Picamera2.")
                return PicameraBackend("picamera2")
            elif OPENCV_AVAILABLE:
                print("[VEDARA] Operator, RPi without Picamera2. Using OpenCV.")
                return OpenCVCamera("opencv:0")

        elif platform == PLATFORM_WINDOWS:
            if OPENCV_AVAILABLE:
                print("[VEDARA] Operator, Windows detected. Using OpenCV.")
                return OpenCVCamera("opencv:0")


        if OPENCV_AVAILABLE:
            print("[VEDARA] Operator, using OpenCV as universal fallback.")
            return OpenCVCamera("opencv:0")


        if use_null_if_unavailable:
            print("[VEDARA] Warning: No camera hardware available. Using NullCamera.")
            return NullCamera()

        raise RuntimeError("No camera backend available. Install opencv-python-headless or picamera2.")

    @staticmethod
    def _create_by_name(name: str) -> CameraInterface:

        name_lower = name.lower()

        if name_lower == "null" or name_lower == "test":
            return NullCamera()

        if name_lower.startswith("picamera"):
            if not PICAMERA2_AVAILABLE:
                raise RuntimeError("Picamera2 not available on this platform")
            return PicameraBackend(name)

        if name_lower.startswith("opencv"):
            if not OPENCV_AVAILABLE:
                raise RuntimeError("OpenCV not available. Install opencv-python-headless")
            return OpenCVCamera(name)

        raise ValueError(f"Unknown camera backend: {name}")

    @staticmethod
    def get_available_backends() -> dict:

        return {
            "opencv": OPENCV_AVAILABLE,
            "picamera2": PICAMERA2_AVAILABLE,
            "null": True
        }

    @staticmethod
    def get_recommended_backend() -> str:

        platform = get_platform()

        if platform == PLATFORM_RPI and PICAMERA2_AVAILABLE:
            return "picamera2"
        if OPENCV_AVAILABLE:
            return "opencv:0"
        return "null"


def create_camera(
    backend_override: Optional[str] = None,
    use_null_if_unavailable: bool = True
) -> CameraInterface:

    return CameraFactory.create(backend_override, use_null_if_unavailable)


__all__ = [

    "CameraInterface",
    "CameraState",
    "CameraProperties",
    "FrameData",


    "OpenCVCamera",
    "PicameraBackend",
    "NullCamera",


    "CameraFactory",
    "create_camera",


    "OPENCV_AVAILABLE",
    "PICAMERA2_AVAILABLE",
]


if __name__ == "__main__":
    import time
    import json

    print("=" * 60)
    print("VEDARA Camera Factory - Integration Test")
    print("=" * 60)


    print("\nAvailable Backends:")
    backends = CameraFactory.get_available_backends()
    for name, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {name}: {status}")


    recommended = CameraFactory.get_recommended_backend()
    print(f"\nRecommended backend: {recommended}")


    print("\n" + "-" * 60)
    print("Creating camera with auto-detection...")

    try:
        camera = create_camera()
        print(f"Camera type: {type(camera).__name__}")
        print(f"Initial state: {camera.state.name}")


        print("\nInitializing camera...")
        success = camera.initialize(width=640, height=480, fps=30)

        if success:
            print(f"Initialization: SUCCESS")
            print(f"Properties: {camera.properties.width}x{camera.properties.height} @ {camera.properties.fps}fps")
            print(f"Backend: {camera.properties.backend_name}")


            camera.start_streaming()


            print("\nCapturing 5 test frames...")
            for i in range(5):
                frame_data = camera.capture_frame()
                if frame_data:
                    print(f"  Frame {frame_data.frame_number}: {frame_data.width}x{frame_data.height} - Valid: {frame_data.is_valid}")
                time.sleep(0.1)


            print("\nCamera Diagnostics:")
            print(json.dumps(camera.get_diagnostics(), indent=2, default=str))


            camera.release()
            print(f"\nFinal state: {camera.state.name}")
        else:
            print(f"Initialization FAILED: {camera.last_error}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Camera factory test complete.")
    print("=" * 60)
