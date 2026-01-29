import time
from typing import Optional, Any, Dict
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from src.camera.interface import (
    CameraInterface,
    CameraState,
    CameraProperties,
    FrameData
)


class OpenCVCamera(CameraInterface):


    def __init__(self, backend_string: str = "opencv:0") -> None:

        super().__init__()

        if not OPENCV_AVAILABLE:
            self._set_error("OpenCV not installed. Run: pip install opencv-python-headless")
            return

        self._backend_string = backend_string
        self._device_source = self._parse_backend_string(backend_string)
        self._capture: Optional[cv2.VideoCapture] = None


        self._frame_buffer: Optional[np.ndarray] = None
        self._target_width: int = 640
        self._target_height: int = 480
        self._target_fps: int = 30


        self._last_capture_time: float = 0.0
        self._min_frame_interval: float = 0.0

    def _parse_backend_string(self, backend_string: str) -> Any:

        if not backend_string.startswith("opencv:"):
            return 0

        source = backend_string[7:]


        try:
            return int(source)
        except ValueError:

            return source

    def _select_capture_backend(self) -> int:

        import platform
        system = platform.system().lower()

        if system == "windows":

            return cv2.CAP_DSHOW
        elif system == "linux":

            return cv2.CAP_V4L2
        else:

            return cv2.CAP_ANY

    def initialize(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs: Any
    ) -> bool:

        if not OPENCV_AVAILABLE:
            self._set_error("OpenCV not available")
            return False

        self._target_width = width
        self._target_height = height
        self._target_fps = fps
        self._min_frame_interval = 1.0 / fps


        backend = self._select_capture_backend()

        try:
            if isinstance(self._device_source, int):

                self._capture = cv2.VideoCapture(self._device_source, backend)
            else:

                self._capture = cv2.VideoCapture(self._device_source)

            if not self._capture.isOpened():
                self._set_error(f"Failed to open camera: {self._device_source}")
                return False


            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._capture.set(cv2.CAP_PROP_FPS, fps)


            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)


            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)


            self._frame_buffer = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)


            self._properties = CameraProperties(
                width=actual_width,
                height=actual_height,
                fps=actual_fps if actual_fps > 0 else float(fps),
                backend_name="OpenCV",
                device_id=str(self._device_source),
                supports_autofocus=True,
                supports_auto_exposure=True,
                supports_auto_wb=True,
                pixel_format="BGR",
                buffer_count=1
            )

            self._state = CameraState.INITIALIZED
            return True

        except Exception as e:
            self._set_error(f"Camera initialization error: {str(e)}")
            return False

    def start_streaming(self) -> bool:

        if self._state not in (CameraState.INITIALIZED, CameraState.PAUSED):
            self._set_error(f"Cannot start streaming from state: {self._state.name}")
            return False

        if self._capture is None or not self._capture.isOpened():
            self._set_error("Camera not initialized or was released")
            return False

        self._state = CameraState.STREAMING
        self._last_capture_time = time.perf_counter()
        return True

    def stop_streaming(self) -> bool:

        if self._state != CameraState.STREAMING:
            return True

        self._state = CameraState.PAUSED
        return True

    def capture_frame(self) -> Optional[FrameData]:

        if self._state != CameraState.STREAMING:
            return None

        if self._capture is None:
            return None

        current_time = time.perf_counter()


        try:

            ret, frame = self._capture.read()

            if not ret or frame is None:
                return None


            if frame.shape[0] != self._target_height or frame.shape[1] != self._target_width:
                frame = cv2.resize(
                    frame,
                    (self._target_width, self._target_height),
                    interpolation=cv2.INTER_LINEAR
                )

            self._last_capture_time = current_time
            frame_num = self._increment_frame_count()

            return FrameData(
                frame=frame,
                timestamp=current_time,
                frame_number=frame_num,
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if len(frame.shape) > 2 else 1
            )

        except Exception as e:
            self._set_error(f"Frame capture error: {str(e)}")
            return None

    def release(self) -> None:

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        self._frame_buffer = None
        self._state = CameraState.RELEASED

    def set_exposure(self, value: float) -> bool:

        if self._capture is None:
            return False

        try:

            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            return self._capture.set(cv2.CAP_PROP_EXPOSURE, value)
        except Exception:
            return False

    def set_auto_exposure(self, enabled: bool) -> bool:

        if self._capture is None:
            return False

        try:

            mode = 0.75 if enabled else 0.25
            return self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode)
        except Exception:
            return False

    def set_white_balance(self, value: float) -> bool:

        if self._capture is None:
            return False

        try:
            return self._capture.set(cv2.CAP_PROP_WB_TEMPERATURE, value)
        except Exception:
            return False

    def set_auto_white_balance(self, enabled: bool) -> bool:

        if self._capture is None:
            return False

        try:
            mode = 1 if enabled else 0
            return self._capture.set(cv2.CAP_PROP_AUTO_WB, mode)
        except Exception:
            return False

    def set_focus(self, value: float) -> bool:

        if self._capture is None:
            return False

        try:

            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            return self._capture.set(cv2.CAP_PROP_FOCUS, value)
        except Exception:
            return False

    def set_autofocus(self, enabled: bool) -> bool:

        if self._capture is None:
            return False

        try:
            mode = 1 if enabled else 0
            return self._capture.set(cv2.CAP_PROP_AUTOFOCUS, mode)
        except Exception:
            return False

    def get_diagnostics(self) -> Dict[str, Any]:

        base_diagnostics = super().get_diagnostics()

        opencv_info = {
            "opencv_available": OPENCV_AVAILABLE,
            "backend_string": self._backend_string,
            "device_source": str(self._device_source),
            "capture_active": self._capture is not None and self._capture.isOpened() if self._capture else False,
        }

        if self._capture is not None and self._capture.isOpened():
            opencv_info.update({
                "actual_width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "actual_height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "actual_fps": self._capture.get(cv2.CAP_PROP_FPS),
                "backend_id": int(self._capture.get(cv2.CAP_PROP_BACKEND)),
                "buffer_size": int(self._capture.get(cv2.CAP_PROP_BUFFERSIZE)),
            })

        base_diagnostics["opencv"] = opencv_info
        return base_diagnostics


if __name__ == "__main__":
    print("=" * 60)
    print("VEDARA OpenCV Camera Backend - Diagnostic Test")
    print("=" * 60)

    if not OPENCV_AVAILABLE:
        print("ERROR: OpenCV not installed!")
        print("Run: pip install opencv-python-headless")
        exit(1)

    print(f"OpenCV Version: {cv2.__version__}")


    camera = OpenCVCamera("opencv:0")
    print(f"\nInitial state: {camera.state.name}")


    print("\nInitializing camera (640x480 @ 30fps)...")
    success = camera.initialize(width=640, height=480, fps=30)

    if not success:
        print(f"ERROR: {camera.last_error}")
        print("\nNote: This test requires a connected camera.")
        print("If no camera is available, the test will fail gracefully.")
        exit(1)

    print(f"Initialize: SUCCESS")
    print(f"Actual resolution: {camera.properties.width}x{camera.properties.height}")
    print(f"Actual FPS: {camera.properties.fps}")


    success = camera.start_streaming()
    print(f"\nStart streaming: {'SUCCESS' if success else 'FAILED'}")


    print("\nCapturing 30 frames...")
    start_time = time.perf_counter()
    captured = 0

    for i in range(30):
        frame_data = camera.capture_frame()
        if frame_data is not None:
            captured += 1
            if (i + 1) % 10 == 0:
                print(f"  Frame {frame_data.frame_number}: {frame_data.width}x{frame_data.height}")

    elapsed = time.perf_counter() - start_time
    actual_fps = captured / elapsed if elapsed > 0 else 0

    print(f"\nCapture stats:")
    print(f"  Frames captured: {captured}/30")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Actual FPS: {actual_fps:.1f}")


    print("\nDiagnostics:")
    import json
    print(json.dumps(camera.get_diagnostics(), indent=2))


    camera.release()
    print(f"\nState after release: {camera.state.name}")

    print("=" * 60)
    print("OpenCV backend test complete.")
    print("=" * 60)
