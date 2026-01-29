import time
import gc
from pathlib import Path
from typing import Optional, Deque, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass, field
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[VEDARA] Warning: psutil not installed. Limited performance monitoring.")

from src.utils.platform_detect import get_platform, PLATFORM_RPI


@dataclass
class PerformanceMetrics:

    timestamp: float = 0.0
    fps: float = 0.0
    frame_time_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    thermal_throttled: bool = False
    gc_collections: int = 0


@dataclass
class FrameTimings:

    capture_ms: float = 0.0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    display_ms: float = 0.0
    total_ms: float = 0.0


class PerformanceMonitor:


    def __init__(
        self,
        fps_window_size: int = 30,
        gc_interval_frames: int = 100,
        thermal_check_interval: int = 50
    ) -> None:

        self._platform = get_platform()
        self._fps_window_size = fps_window_size
        self._gc_interval = gc_interval_frames
        self._thermal_interval = thermal_check_interval


        self._frame_times: Deque[float] = deque(maxlen=fps_window_size)
        self._last_frame_time: float = time.perf_counter()
        self._frame_count: int = 0


        self._current_timings: FrameTimings = FrameTimings()
        self._timing_stack: list[tuple[str, float]] = []


        self._cached_metrics: PerformanceMetrics = PerformanceMetrics()
        self._last_metrics_update: float = 0.0
        self._metrics_update_interval: float = 0.25


        self._thermal_path: Optional[Path] = None
        self._last_temperature: float = 0.0
        if self._platform == PLATFORM_RPI:
            self._thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")


        self._gc_count: int = 0


        self._process: Optional[Any] = None
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()

    def frame_start(self) -> None:

        current_time = time.perf_counter()


        if self._last_frame_time > 0:
            frame_delta = current_time - self._last_frame_time
            self._frame_times.append(frame_delta)

        self._last_frame_time = current_time
        self._frame_count += 1


        self._current_timings = FrameTimings()
        self._timing_stack.clear()

    def frame_end(self) -> None:


        self._current_timings.total_ms = (
            time.perf_counter() - self._last_frame_time
        ) * 1000.0


        if self._platform == PLATFORM_RPI:
            if self._frame_count % self._gc_interval == 0:
                gc.collect()
                self._gc_count = sum(gc.get_count())


        current_time = time.perf_counter()
        if current_time - self._last_metrics_update >= self._metrics_update_interval:
            self._update_system_metrics()
            self._last_metrics_update = current_time

    def start_timer(self, name: str) -> None:

        self._timing_stack.append((name, time.perf_counter()))

    def stop_timer(self, name: str) -> float:

        end_time = time.perf_counter()
        elapsed_ms = 0.0


        for i in range(len(self._timing_stack) - 1, -1, -1):
            if self._timing_stack[i][0] == name:
                start_time = self._timing_stack[i][1]
                elapsed_ms = (end_time - start_time) * 1000.0
                self._timing_stack.pop(i)
                break


        if name == "capture":
            self._current_timings.capture_ms = elapsed_ms
        elif name == "inference":
            self._current_timings.inference_ms = elapsed_ms
        elif name == "render":
            self._current_timings.render_ms = elapsed_ms
        elif name == "display":
            self._current_timings.display_ms = elapsed_ms

        return elapsed_ms

    def _update_system_metrics(self) -> None:

        metrics = self._cached_metrics
        metrics.timestamp = time.time()


        if len(self._frame_times) > 0:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            metrics.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            metrics.frame_time_ms = avg_frame_time * 1000.0


        if PSUTIL_AVAILABLE and self._process is not None:
            try:
                metrics.cpu_percent = self._process.cpu_percent()
                mem_info = self._process.memory_info()
                metrics.memory_used_mb = mem_info.rss / (1024 * 1024)
                metrics.memory_percent = self._process.memory_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


        if self._platform == PLATFORM_RPI:
            metrics.temperature_c = self._read_temperature()
            metrics.thermal_throttled = metrics.temperature_c >= 70.0


        metrics.gc_collections = self._gc_count

    def _read_temperature(self) -> float:

        if self._thermal_path is None or not self._thermal_path.exists():
            return 0.0

        try:
            temp_str = self._thermal_path.read_text().strip()

            temp_c = int(temp_str) / 1000.0
            self._last_temperature = temp_c
            return temp_c
        except (ValueError, IOError, PermissionError):
            return self._last_temperature

    def get_fps(self) -> float:

        if len(self._frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_metrics(self) -> PerformanceMetrics:

        return self._cached_metrics

    def get_timings(self) -> FrameTimings:

        return self._current_timings

    def get_frame_count(self) -> int:

        return self._frame_count

    def should_skip_frame(self, max_cpu: float = 85.0) -> bool:

        metrics = self._cached_metrics


        if metrics.thermal_throttled:
            return True


        if metrics.cpu_percent > max_cpu:
            return True

        return False

    def force_gc(self) -> None:

        gc.collect()
        self._gc_count = sum(gc.get_count())

    def get_status_string(self) -> str:

        metrics = self._cached_metrics

        status_parts = [
            f"FPS: {metrics.fps:.1f}",
            f"CPU: {metrics.cpu_percent:.0f}%",
            f"RAM: {metrics.memory_used_mb:.0f}MB"
        ]

        if self._platform == PLATFORM_RPI and metrics.temperature_c > 0:
            status_parts.append(f"TEMP: {metrics.temperature_c:.0f}°C")
            if metrics.thermal_throttled:
                status_parts.append("⚠ THROTTLED")

        return " | ".join(status_parts)

    def get_detailed_report(self) -> Dict[str, Any]:

        metrics = self._cached_metrics
        timings = self._current_timings

        return {
            "frame_count": self._frame_count,
            "fps": {
                "current": round(metrics.fps, 2),
                "target_min": 18,
                "target_max": 22,
                "in_target": 18 <= metrics.fps <= 22
            },
            "frame_time": {
                "total_ms": round(timings.total_ms, 2),
                "capture_ms": round(timings.capture_ms, 2),
                "inference_ms": round(timings.inference_ms, 2),
                "render_ms": round(timings.render_ms, 2),
                "display_ms": round(timings.display_ms, 2)
            },
            "system": {
                "cpu_percent": round(metrics.cpu_percent, 1),
                "memory_mb": round(metrics.memory_used_mb, 1),
                "memory_percent": round(metrics.memory_percent, 1)
            },
            "thermal": {
                "temperature_c": round(metrics.temperature_c, 1),
                "throttled": metrics.thermal_throttled
            },
            "gc_collections": metrics.gc_collections,
            "platform": self._platform
        }


_monitor_instance: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:

    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance


class TimerContext:


    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor or get_monitor()
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> 'TimerContext':
        self.monitor.start_timer(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = self.monitor.stop_timer(self.name)


if __name__ == "__main__":
    import random

    print("=" * 60)
    print("VEDARA Performance Monitor - Diagnostic Test")
    print("=" * 60)

    monitor = get_monitor()

    print("\nSimulating 100 frames of processing...")
    print("-" * 60)

    for i in range(100):

        monitor.frame_start()


        with TimerContext("capture"):
            time.sleep(random.uniform(0.005, 0.010))


        with TimerContext("inference"):
            time.sleep(random.uniform(0.020, 0.035))


        with TimerContext("render"):
            time.sleep(random.uniform(0.003, 0.008))


        monitor.frame_end()


        if (i + 1) % 25 == 0:
            print(f"Frame {i + 1}: {monitor.get_status_string()}")

    print("\n" + "=" * 60)
    print("Final Performance Report:")
    print("=" * 60)

    import json
    report = monitor.get_detailed_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"Total frames processed: {monitor.get_frame_count()}")
    print(f"Final FPS: {monitor.get_fps():.2f}")
    print("=" * 60)
