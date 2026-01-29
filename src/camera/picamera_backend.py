import time
from typing import Optional, Any, Dict
import numpy as np


try:
    from picamera2 import Picamera2
    from picamera2.encoders import Encoder
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None
    controls = None

from src.camera.interface import (
    CameraInterface,
    CameraState,
    CameraProperties,
    FrameData
)


class PicameraBackend(CameraInterface):


    def __init__(self, backend_string: str = "picamera2") -> None:

        super().__init__()

        if not PICAMERA2_AVAILABLE:
            self._set_error("Picamera2 not installed. Run: pip install picamera2")
            return

        self._backend_string = backend_string
        self._camera_num = self._parse_backend_string(backend_string)
        self._picam: Optional[Picamera2] = None


        self._frame_buffer: Optional[np.ndarray] = None
        self._target_width: int = 640
        self._target_height: int = 480
        self._target_fps: int = 30


        self._config: Optional[Dict] = None

    def _parse_backend_string(self, backend_string: str) -> int:

        if ":" in backend_string:
            try:
                return int(backend_string.split(":")[1])
            except (ValueError, IndexError):
                return 0
        return 0

    def initialize(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs: Any
    ) -> bool:

        if not PICAMERA2_AVAILABLE:
            self._set_error("Picamera2 not available on this platform")
            return False

        self._target_width = width
        self._target_height = height
        self._target_fps = fps

        try:

            self._picam = Picamera2(camera_num=self._camera_num)


            self._config = self._picam.create_preview_configuration(
                main={
                    "size": (width, height),
                    "format": "RGB888"
                },
                buffer_count=kwargs.get("buffer_count", 4),
                queue=True,
                controls={
                    "FrameDurationLimits": (
                        int(1000000 / fps),
                        int(1000000 / fps)
                    )
                }
            )


            self._picam.align_configuration(self._config)


            self._picam.configure(self._config)


            self._frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)


            actual_size = self._config["main"]["size"]

            self._properties = CameraProperties(
                width=actual_size[0],
                height=actual_size[1],
                fps=float(fps),
                backend_name="Picamera2",
                device_id=f"camera{self._camera_num}",
                supports_autofocus=self._check_autofocus_support(),
                supports_auto_exposure=True,
                supports_auto_wb=True,
                pixel_format="RGB",
                buffer_count=self._config.get("buffer_count", 4)
            )

            self._state = CameraState.INITIALIZED
            return True

        except Exception as e:
            self._set_error(f"Picamera2 initialization error: {str(e)}")
            if self._picam is not None:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None
            return False

    def _check_autofocus_support(self) -> bool:

        if self._picam is None:
            return False

        try:
            camera_properties = self._picam.camera_properties

            return "AfMode" in str(camera_properties)
        except Exception:
            return False

    def start_streaming(self) -> bool:

        if self._state not in (CameraState.INITIALIZED, CameraState.PAUSED):
            self._set_error(f"Cannot start streaming from state: {self._state.name}")
            return False

        if self._picam is None:
            self._set_error("Camera not initialized")
            return False

        try:
            self._picam.start()
            self._state = CameraState.STREAMING
            return True
        except Exception as e:
            self._set_error(f"Failed to start streaming: {str(e)}")
            return False

    def stop_streaming(self) -> bool:

        if self._state != CameraState.STREAMING:
            return True

        if self._picam is None:
            return True

        try:
            self._picam.stop()
            self._state = CameraState.PAUSED
            return True
        except Exception as e:
            self._set_error(f"Failed to stop streaming: {str(e)}")
            return False

    def capture_frame(self) -> Optional[FrameData]:

        if self._state != CameraState.STREAMING:
            return None

        if self._picam is None:
            return None

        try:


            frame = self._picam.capture_array("main")

            if frame is None:
                return None


            frame_bgr = frame[:, :, ::-1].copy()

            current_time = time.perf_counter()
            frame_num = self._increment_frame_count()

            return FrameData(
                frame=frame_bgr,
                timestamp=current_time,
                frame_number=frame_num,
                width=frame_bgr.shape[1],
                height=frame_bgr.shape[0],
                channels=3
            )

        except Exception as e:
            self._set_error(f"Frame capture error: {str(e)}")
            return None

    def release(self) -> None:

        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass

            try:
                self._picam.close()
            except Exception:
                pass

            self._picam = None

        self._frame_buffer = None
        self._state = CameraState.RELEASED

    def set_exposure(self, value: float) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        try:
            self._picam.set_controls({
                "ExposureTime": int(value),
                "AeEnable": False
            })
            return True
        except Exception:
            return False

    def set_auto_exposure(self, enabled: bool) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        try:
            self._picam.set_controls({"AeEnable": enabled})
            return True
        except Exception:
            return False

    def set_white_balance(self, value: float) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        try:


            if value < 4000:
                red_gain = 1.0
                blue_gain = 1.0 + (4000 - value) / 2000
            else:
                red_gain = 1.0 + (value - 4000) / 4000
                blue_gain = 1.0

            self._picam.set_controls({
                "AwbEnable": False,
                "ColourGains": (red_gain, blue_gain)
            })
            return True
        except Exception:
            return False

    def set_auto_white_balance(self, enabled: bool) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        try:
            self._picam.set_controls({"AwbEnable": enabled})
            return True
        except Exception:
            return False

    def set_autofocus(self, enabled: bool) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        if not self._properties.supports_autofocus:
            return False

        try:
            if enabled:
                self._picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            else:
                self._picam.set_controls({"AfMode": controls.AfModeEnum.Manual})
            return True
        except Exception:
            return False

    def set_focus(self, value: float) -> bool:

        if self._picam is None or not PICAMERA2_AVAILABLE:
            return False

        if not self._properties.supports_autofocus:
            return False

        try:
            self._picam.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": value
            })
            return True
        except Exception:
            return False

    def get_diagnostics(self) -> Dict[str, Any]:

        base_diagnostics = super().get_diagnostics()

        picam_info = {
            "picamera2_available": PICAMERA2_AVAILABLE,
            "backend_string": self._backend_string,
            "camera_num": self._camera_num,
            "camera_active": self._picam is not None,
        }

        if self._picam is not None:
            try:
                camera_props = self._picam.camera_properties
                picam_info.update({
                    "model": camera_props.get("Model", "Unknown"),
                    "sensor_size": str(camera_props.get("PixelArraySize", "Unknown")),
                })


                if self._state == CameraState.STREAMING:
                    metadata = self._picam.capture_metadata()
                    picam_info.update({
                        "exposure_time": metadata.get("ExposureTime", 0),
                        "analogue_gain": metadata.get("AnalogueGain", 0),
                        "colour_temp": metadata.get("ColourTemperature", 0),
                    })
            except Exception:
                pass

        base_diagnostics["picamera2"] = picam_info
        return base_diagnostics


if __name__ == "__main__":
    print("=" * 60)
    print("VEDARA Picamera2 Backend - Diagnostic Test")
    print("=" * 60)

    if not PICAMERA2_AVAILABLE:
        print("\nPicamera2 not available on this platform.")
        print("This module is designed for Raspberry Pi with camera module.")
        print("\nOn Raspberry Pi, install with:")
        print("  sudo apt install -y python3-picamera2")
        print("  pip install picamera2")
        print("\nFor Windows development, use OpenCV backend instead.")
        print("=" * 60)
        exit(0)


    camera = PicameraBackend("picamera2")
    print(f"\nInitial state: {camera.state.name}")


    print("\nInitializing camera (640x480 @ 30fps)...")
    success = camera.initialize(width=640, height=480, fps=30)

    if not success:
        print(f"ERROR: {camera.last_error}")
        exit(1)

    print(f"Initialize: SUCCESS")
    print(f"Resolution: {camera.properties.width}x{camera.properties.height}")
    print(f"Autofocus: {'Supported' if camera.properties.supports_autofocus else 'Not supported'}")


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
    diag = camera.get_diagnostics()
    print(json.dumps(diag, indent=2, default=str))


    camera.release()
    print(f"\nState after release: {camera.state.name}")

    print("=" * 60)
    print("Picamera2 backend test complete.")
    print("=" * 60)
