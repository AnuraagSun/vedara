import time
import gc
import signal
import sys
import numpy as np
import cv2
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from src.camera import create_camera, CameraInterface, FrameData
from src.inference import create_detector, DetectorInterface, InferenceResult
from src.hud import get_hud_renderer, HUDRenderer
from src.utils.platform_detect import get_platform, PLATFORM_RPI, PLATFORM_WINDOWS
from src.utils.config_loader import get_config
from src.utils.performance import get_monitor, PerformanceMonitor, TimerContext


class EngineState(Enum):

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class EngineStats:

    total_frames: int = 0
    total_detections: int = 0
    start_time: float = 0.0
    run_time: float = 0.0
    avg_fps: float = 0.0
    avg_inference_ms: float = 0.0
    avg_render_ms: float = 0.0
    frames_skipped: int = 0
    gc_runs: int = 0


@dataclass
class EngineConfig:


    window_name: str = "VEDARA AR System"
    fullscreen: bool = False
    show_preview: bool = True


    target_fps: int = 20
    max_cpu_percent: int = 85
    enable_frame_skip: bool = True
    gc_interval: int = 100


    enable_hud: bool = True
    enable_detection: bool = True
    save_frames: bool = False
    save_path: Path = field(default_factory=lambda: Path("output"))


    on_detection: Optional[Callable[[InferenceResult], None]] = None
    on_frame: Optional[Callable[[np.ndarray, InferenceResult], None]] = None


class VedaraEngine:


    def __init__(self, config: Optional[EngineConfig] = None) -> None:

        self._config = config or EngineConfig()
        self._state = EngineState.UNINITIALIZED
        self._platform = get_platform()


        self._camera: Optional[CameraInterface] = None
        self._detector: Optional[DetectorInterface] = None
        self._hud: Optional[HUDRenderer] = None
        self._perf_monitor: Optional[PerformanceMonitor] = None


        self._stats = EngineStats()
        self._last_result: Optional[InferenceResult] = None
        self._running = False
        self._frame_interval = 1.0 / self._config.target_fps


        self._frame_buffer: Optional[np.ndarray] = None


        self._setup_signal_handlers()


        self._last_error: Optional[str] = None

    def _setup_signal_handlers(self) -> None:

        def signal_handler(signum, frame):
            print("\n[VEDARA] Operator, shutdown signal received.")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @property
    def state(self) -> EngineState:

        return self._state

    @property
    def stats(self) -> EngineStats:

        return self._stats

    @property
    def is_running(self) -> bool:

        return self._running and self._state == EngineState.RUNNING

    @property
    def last_error(self) -> Optional[str]:

        return self._last_error

    def initialize(self) -> bool:

        print("=" * 60)
        print("VEDARA AR System - Initialization")
        print("=" * 60)

        self._state = EngineState.INITIALIZING
        app_config = get_config()


        print(f"\n[1/5] Platform Detection")
        print(f"      Platform: {self._platform}")
        print(f"      Target FPS: {self._config.target_fps}")


        print(f"\n[2/5] Camera Initialization")
        try:
            self._camera = create_camera(use_null_if_unavailable=True)
            success = self._camera.initialize(
                width=app_config.camera.width,
                height=app_config.camera.height,
                fps=app_config.camera.target_fps
            )

            if not success:
                self._last_error = f"Camera init failed: {self._camera.last_error}"
                print(f"      [ERROR] {self._last_error}")
                self._state = EngineState.ERROR
                return False

            print(f"      Backend: {self._camera.properties.backend_name}")
            print(f"      Resolution: {self._camera.properties.width}x{self._camera.properties.height}")
            print(f"      [OK] Camera ready")

        except Exception as e:
            self._last_error = f"Camera exception: {str(e)}"
            print(f"      [ERROR] {self._last_error}")
            self._state = EngineState.ERROR
            return False


        print(f"\n[3/5] Detector Initialization")
        try:
            self._detector = create_detector(use_mock_if_unavailable=True)
            print(f"      Model: {self._detector.model_name}")
            print(f"      Input: {self._detector.input_size}")
            print(f"      [OK] Detector ready")

        except Exception as e:
            self._last_error = f"Detector exception: {str(e)}"
            print(f"      [ERROR] {self._last_error}")
            self._state = EngineState.ERROR
            return False


        print(f"\n[4/5] HUD Initialization")
        try:
            self._hud = get_hud_renderer()
            self._hud.set_enabled(self._config.enable_hud)
            print(f"      Enabled: {self._hud.is_enabled()}")
            print(f"      [OK] HUD ready")

        except Exception as e:
            self._last_error = f"HUD exception: {str(e)}"
            print(f"      [ERROR] {self._last_error}")
            self._state = EngineState.ERROR
            return False


        print(f"\n[5/5] Performance Monitor")
        self._perf_monitor = get_monitor()
        print(f"      GC Interval: {self._config.gc_interval} frames")
        print(f"      Frame Skip: {'Enabled' if self._config.enable_frame_skip else 'Disabled'}")
        print(f"      [OK] Monitor ready")


        cam_props = self._camera.properties
        self._frame_buffer = np.zeros(
            (cam_props.height, cam_props.width, 3),
            dtype=np.uint8
        )
        print(f"\n[RPi OPT] Frame buffer pre-allocated: {cam_props.width}x{cam_props.height}")


        self._state = EngineState.READY
        print("\n" + "=" * 60)
        print("Initialization complete. All systems nominal.")
        print("=" * 60)

        return True

    def run(self) -> None:

        if self._state != EngineState.READY:
            print("[VEDARA] Error: Engine not initialized. Call initialize() first.")
            return

        print("\n[VEDARA] Starting main loop...")
        print("[VEDARA] Press 'Q' to quit, 'H' to toggle HUD, 'D' to toggle detection")
        print("-" * 60)


        if not self._camera.start_streaming():
            self._last_error = "Failed to start camera streaming"
            print(f"[VEDARA] Error: {self._last_error}")
            return


        self._state = EngineState.RUNNING
        self._running = True
        self._stats.start_time = time.perf_counter()


        if self._config.show_preview:
            try:
                cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
                if self._config.fullscreen:
                    cv2.setWindowProperty(
                        self._config.window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )
            except Exception as e:
                print(f"[VEDARA] Warning: Cannot create display window: {e}")
                self._config.show_preview = False


        try:
            self._main_loop()
        except Exception as e:
            self._last_error = f"Main loop exception: {str(e)}"
            print(f"[VEDARA] Error: {self._last_error}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _main_loop(self) -> None:

        frame_count = 0
        inference_times = []
        render_times = []
        skip_inference = False

        while self._running:
            loop_start = time.perf_counter()


            self._perf_monitor.frame_start()


            with TimerContext("capture", self._perf_monitor):
                frame_data = self._camera.capture_frame()

            if frame_data is None or not frame_data.is_valid:
                continue

            frame = frame_data.frame
            frame_count += 1
            self._stats.total_frames = frame_count


            if self._config.enable_frame_skip:
                skip_inference = self._perf_monitor.should_skip_frame(
                    self._config.max_cpu_percent
                )
                if skip_inference:
                    self._stats.frames_skipped += 1


            result = self._last_result
            if self._config.enable_detection and not skip_inference:
                with TimerContext("inference", self._perf_monitor):
                    result = self._detector.detect(frame)
                    self._last_result = result
                    self._stats.total_detections += result.detection_count
                    inference_times.append(result.inference_time_ms)


                if self._config.on_detection and result.detection_count > 0:
                    self._config.on_detection(result)


            if self._config.enable_hud:
                with TimerContext("render", self._perf_monitor):
                    frame = self._hud.render(
                        frame,
                        detections=result,
                        show_performance=True,
                        show_title=True
                    )
                    render_times.append(
                        self._perf_monitor.get_timings().render_ms
                    )


            if self._config.on_frame:
                self._config.on_frame(frame, result)


            if self._config.show_preview:
                with TimerContext("display", self._perf_monitor):
                    try:
                        cv2.imshow(self._config.window_name, frame)


                        key = cv2.waitKey(1) & 0xFF
                        if not self._handle_key(key):
                            break
                    except Exception:
                        self._config.show_preview = False


            if self._config.save_frames and frame_count % 30 == 0:
                self._save_frame(frame, frame_count)


            self._perf_monitor.frame_end()


            if self._platform == PLATFORM_RPI:
                if frame_count % self._config.gc_interval == 0:
                    gc.collect()
                    self._stats.gc_runs += 1


            loop_time = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)


            if frame_count % 60 == 0:
                metrics = self._perf_monitor.get_metrics()
                det_count = result.detection_count if result else 0
                print(f"[VEDARA] Frame {frame_count}: "
                      f"FPS={metrics.fps:.1f}, "
                      f"Det={det_count}, "
                      f"CPU={metrics.cpu_percent:.0f}%")


        self._stats.run_time = time.perf_counter() - self._stats.start_time
        self._stats.avg_fps = self._stats.total_frames / self._stats.run_time if self._stats.run_time > 0 else 0
        self._stats.avg_inference_ms = sum(inference_times) / len(inference_times) if inference_times else 0
        self._stats.avg_render_ms = sum(render_times) / len(render_times) if render_times else 0

    def _handle_key(self, key: int) -> bool:

        if key == ord('q') or key == 27:
            print("[VEDARA] Operator, quit command received.")
            return False

        elif key == ord('h'):

            self._config.enable_hud = not self._config.enable_hud
            self._hud.set_enabled(self._config.enable_hud)
            print(f"[VEDARA] HUD: {'ON' if self._config.enable_hud else 'OFF'}")

        elif key == ord('d'):

            self._config.enable_detection = not self._config.enable_detection
            print(f"[VEDARA] Detection: {'ON' if self._config.enable_detection else 'OFF'}")

        elif key == ord('s'):

            self._save_frame(None, self._stats.total_frames, screenshot=True)

        elif key == ord('f'):

            self._config.fullscreen = not self._config.fullscreen
            if self._config.fullscreen:
                cv2.setWindowProperty(
                    self._config.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
            else:
                cv2.setWindowProperty(
                    self._config.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )

        return True

    def _save_frame(
        self,
        frame: Optional[np.ndarray],
        frame_num: int,
        screenshot: bool = False
    ) -> None:

        self._config.save_path.mkdir(parents=True, exist_ok=True)

        if screenshot:
            filename = self._config.save_path / f"screenshot_{frame_num:06d}.png"
        else:
            filename = self._config.save_path / f"frame_{frame_num:06d}.jpg"

        if frame is not None:
            cv2.imwrite(str(filename), frame)
            print(f"[VEDARA] Saved: {filename}")

    def stop(self) -> None:

        if self._running:
            print("[VEDARA] Stopping engine...")
            self._running = False
            self._state = EngineState.STOPPING

    def _cleanup(self) -> None:

        print("\n[VEDARA] Cleaning up...")


        if self._camera is not None:
            self._camera.release()
            print("  Camera released")


        if self._detector is not None:
            self._detector.release()
            print("  Detector released")


        try:
            cv2.destroyAllWindows()
            print("  Display closed")
        except Exception:
            pass

        self._state = EngineState.STOPPED
        print("[VEDARA] Cleanup complete")

    def shutdown(self) -> None:

        self.stop()
        self._cleanup()
        self._print_final_stats()

    def _print_final_stats(self) -> None:

        stats = self._stats

        print("\n" + "=" * 60)
        print("VEDARA AR System - Session Statistics")
        print("=" * 60)
        print(f"  Total Frames:      {stats.total_frames}")
        print(f"  Total Detections:  {stats.total_detections}")
        print(f"  Run Time:          {stats.run_time:.1f}s")
        print(f"  Average FPS:       {stats.avg_fps:.1f}")
        print(f"  Avg Inference:     {stats.avg_inference_ms:.1f}ms")
        print(f"  Avg Render:        {stats.avg_render_ms:.1f}ms")
        print(f"  Frames Skipped:    {stats.frames_skipped}")
        print(f"  GC Runs:           {stats.gc_runs}")
        print("=" * 60)

    def get_diagnostics(self) -> Dict[str, Any]:

        return {
            "state": self._state.name,
            "platform": self._platform,
            "running": self._running,
            "stats": {
                "total_frames": self._stats.total_frames,
                "total_detections": self._stats.total_detections,
                "avg_fps": round(self._stats.avg_fps, 2),
                "frames_skipped": self._stats.frames_skipped,
            },
            "camera": self._camera.get_diagnostics() if self._camera else None,
            "detector": self._detector.get_diagnostics() if self._detector else None,
            "last_error": self._last_error,
        }


def create_engine(config: Optional[EngineConfig] = None) -> VedaraEngine:

    return VedaraEngine(config)


def run_vedara(
    show_preview: bool = True,
    enable_detection: bool = True,
    enable_hud: bool = True,
    fullscreen: bool = False
) -> None:

    config = EngineConfig(
        show_preview=show_preview,
        enable_detection=enable_detection,
        enable_hud=enable_hud,
        fullscreen=fullscreen
    )

    engine = VedaraEngine(config)

    if engine.initialize():
        engine.run()

    engine.shutdown()
