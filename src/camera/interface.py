from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class CameraState(Enum):

    UNINITIALIZED = auto()
    INITIALIZED = auto()
    STREAMING = auto()
    PAUSED = auto()
    ERROR = auto()
    RELEASED = auto()


@dataclass
class CameraProperties:

    width: int = 640
    height: int = 480
    fps: float = 30.0
    backend_name: str = "unknown"
    device_id: str = "unknown"
    supports_autofocus: bool = False
    supports_auto_exposure: bool = True
    supports_auto_wb: bool = True
    pixel_format: str = "BGR"
    buffer_count: int = 4


@dataclass
class FrameData:

    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    channels: int = 3

    @property
    def shape(self) -> Tuple[int, int, int]:

        return (self.height, self.width, self.channels)

    @property
    def is_valid(self) -> bool:

        if self.frame is None:
            return False
        if self.frame.shape[0] != self.height:
            return False
        if self.frame.shape[1] != self.width:
            return False
        return True


class CameraInterface(ABC):


    def __init__(self) -> None:

        self._state: CameraState = CameraState.UNINITIALIZED
        self._properties: CameraProperties = CameraProperties()
        self._frame_count: int = 0
        self._last_error: Optional[str] = None

    @property
    def state(self) -> CameraState:

        return self._state

    @property
    def properties(self) -> CameraProperties:

        return self._properties

    @property
    def frame_count(self) -> int:

        return self._frame_count

    @property
    def last_error(self) -> Optional[str]:

        return self._last_error

    @property
    def is_ready(self) -> bool:

        return self._state == CameraState.STREAMING

    @abstractmethod
    def initialize(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs: Any
    ) -> bool:

        pass

    @abstractmethod
    def start_streaming(self) -> bool:

        pass

    @abstractmethod
    def stop_streaming(self) -> bool:

        pass

    @abstractmethod
    def capture_frame(self) -> Optional[FrameData]:

        pass

    @abstractmethod
    def release(self) -> None:

        pass

    @abstractmethod
    def set_exposure(self, value: float) -> bool:

        pass

    @abstractmethod
    def set_auto_exposure(self, enabled: bool) -> bool:

        pass

    @abstractmethod
    def set_white_balance(self, value: float) -> bool:

        pass

    @abstractmethod
    def set_auto_white_balance(self, enabled: bool) -> bool:

        pass

    def get_diagnostics(self) -> Dict[str, Any]:

        return {
            "state": self._state.name,
            "frame_count": self._frame_count,
            "last_error": self._last_error,
            "properties": {
                "width": self._properties.width,
                "height": self._properties.height,
                "fps": self._properties.fps,
                "backend": self._properties.backend_name,
                "device": self._properties.device_id,
            }
        }

    def _set_error(self, message: str) -> None:

        self._last_error = message
        self._state = CameraState.ERROR

    def _increment_frame_count(self) -> int:

        self._frame_count += 1
        return self._frame_count


class NullCamera(CameraInterface):


    def __init__(self) -> None:
        super().__init__()
        self._synthetic_frame: Optional[np.ndarray] = None
        self._width: int = 640
        self._height: int = 480

    def initialize(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs: Any
    ) -> bool:

        self._width = width
        self._height = height


        self._synthetic_frame = np.zeros((height, width, 3), dtype=np.uint8)


        self._generate_test_pattern()

        self._properties = CameraProperties(
            width=width,
            height=height,
            fps=float(fps),
            backend_name="NullCamera",
            device_id="synthetic",
            supports_autofocus=False,
            supports_auto_exposure=False,
            supports_auto_wb=False,
            pixel_format="BGR",
            buffer_count=1
        )

        self._state = CameraState.INITIALIZED
        return True

    def _generate_test_pattern(self) -> None:

        if self._synthetic_frame is None:
            return


        self._synthetic_frame[:] = [25, 10, 15]


        for y in range(0, self._height, 40):
            self._synthetic_frame[y:y+1, :] = [255, 243, 0]
        for x in range(0, self._width, 40):
            self._synthetic_frame[:, x:x+1] = [255, 243, 0]


        cx, cy = self._width // 2, self._height // 2
        self._synthetic_frame[cy-20:cy+20, cx-1:cx+2] = [175, 42, 183]
        self._synthetic_frame[cy-1:cy+2, cx-20:cx+20] = [175, 42, 183]

    def start_streaming(self) -> bool:

        if self._state not in (CameraState.INITIALIZED, CameraState.PAUSED):
            self._set_error("Cannot start streaming: invalid state")
            return False

        self._state = CameraState.STREAMING
        return True

    def stop_streaming(self) -> bool:

        self._state = CameraState.PAUSED
        return True

    def capture_frame(self) -> Optional[FrameData]:

        if self._state != CameraState.STREAMING:
            return None

        if self._synthetic_frame is None:
            return None

        import time
        frame_num = self._increment_frame_count()


        animated_frame = self._synthetic_frame.copy()


        scanline_y = (frame_num * 3) % self._height
        animated_frame[scanline_y:scanline_y+2, :] = [0, 255, 255]

        return FrameData(
            frame=animated_frame,
            timestamp=time.perf_counter(),
            frame_number=frame_num,
            width=self._width,
            height=self._height,
            channels=3
        )

    def release(self) -> None:

        self._synthetic_frame = None
        self._state = CameraState.RELEASED

    def set_exposure(self, value: float) -> bool:

        return True

    def set_auto_exposure(self, enabled: bool) -> bool:

        return True

    def set_white_balance(self, value: float) -> bool:

        return True

    def set_auto_white_balance(self, enabled: bool) -> bool:

        return True


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("VEDARA Camera Interface - Null Camera Test")
    print("=" * 60)


    camera = NullCamera()
    print(f"\nInitial state: {camera.state.name}")


    success = camera.initialize(width=640, height=480, fps=30)
    print(f"Initialize: {'SUCCESS' if success else 'FAILED'}")
    print(f"State after init: {camera.state.name}")
    print(f"Properties: {camera.properties.width}x{camera.properties.height} @ {camera.properties.fps}fps")


    success = camera.start_streaming()
    print(f"\nStart streaming: {'SUCCESS' if success else 'FAILED'}")
    print(f"State after start: {camera.state.name}")


    print("\nCapturing 10 frames...")
    for i in range(10):
        frame_data = camera.capture_frame()
        if frame_data is not None:
            print(f"  Frame {frame_data.frame_number}: {frame_data.width}x{frame_data.height}, valid={frame_data.is_valid}")
        time.sleep(0.033)


    print("\nDiagnostics:")
    import json
    print(json.dumps(camera.get_diagnostics(), indent=2))


    camera.release()
    print(f"\nState after release: {camera.state.name}")

    print("=" * 60)
    print("Null camera test complete. Interface contract validated.")
    print("=" * 60)
