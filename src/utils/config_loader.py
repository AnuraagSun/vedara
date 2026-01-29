import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy

from src.utils.platform_detect import (
    get_detector,
    get_project_root,
    get_platform,
    PLATFORM_RPI,
    PLATFORM_WINDOWS,
    PlatformType
)


@dataclass
class CameraConfig:

    width: int = 640
    height: int = 480
    target_fps: int = 20
    backend: str = "opencv:0"
    buffer_count: int = 4
    auto_exposure: bool = True
    auto_wb: bool = True


@dataclass
class InferenceConfig:

    model_file: str = "yolov5n_int8.tflite"
    model_path: Path = field(default_factory=Path)
    input_width: int = 320
    input_height: int = 320
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    max_detections: int = 20
    threads: int = 4


@dataclass
class GlowConfig:

    enabled: bool = True
    layers: int = 3
    outer_thickness: int = 8
    middle_thickness: int = 4
    inner_thickness: int = 2
    outer_alpha: float = 0.15
    middle_alpha: float = 0.40
    inner_alpha: float = 1.0


@dataclass
class HUDConfig:

    enabled: bool = True
    colors: Dict[str, Any] = field(default_factory=dict)
    glow: GlowConfig = field(default_factory=GlowConfig)
    scanlines_enabled: bool = True
    scanlines_opacity: float = 0.08
    scanlines_spacing: int = 4
    animations_enabled: bool = True
    slide_duration_ms: int = 150
    fade_duration_ms: int = 100
    font_scale: float = 0.6
    font_thickness: int = 1
    outline_thickness: int = 2


@dataclass
class PerformanceConfig:

    min_fps: int = 18
    max_fps: int = 22
    gc_interval_frames: int = 100
    preallocate_buffers: bool = True
    thermal_enabled: bool = True
    throttle_temp_c: int = 70
    critical_temp_c: int = 80
    thermal_check_interval: int = 50
    max_cpu_percent: int = 85
    adaptive_frameskip: bool = True


@dataclass
class DisplayConfig:

    title: str = "VEDARA AR System"
    fullscreen: bool = False
    output_width: int = 640
    output_height: int = 480
    show_fps: bool = True
    show_cpu: bool = True
    show_memory: bool = True
    show_temp: bool = True


@dataclass
class DebugConfig:

    simulate_rpi: bool = False
    throttle_cpu_ghz: float = 1.5
    save_frames: bool = False
    save_interval: int = 100
    save_path: Path = field(default_factory=lambda: Path("debug/frames"))
    show_detection_time: bool = True
    show_render_time: bool = True


class ConfigLoader:


    _instance: Optional['ConfigLoader'] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[Path] = None) -> 'ConfigLoader':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None) -> None:

        if ConfigLoader._initialized:
            return

        self._platform: PlatformType = get_platform()
        self._project_root: Path = get_project_root()
        self._raw_config: Dict[str, Any] = {}


        if config_path is None:
            config_path = self._project_root / "config" / "config.yaml"
        self._config_path: Path = config_path.resolve()


        self._load_config()


        self.camera: CameraConfig = self._parse_camera_config()
        self.inference: InferenceConfig = self._parse_inference_config()
        self.hud: HUDConfig = self._parse_hud_config()
        self.performance: PerformanceConfig = self._parse_performance_config()
        self.display: DisplayConfig = self._parse_display_config()
        self.debug: DebugConfig = self._parse_debug_config()

        ConfigLoader._initialized = True

    def _load_config(self) -> None:

        if not self._config_path.exists():
            print(f"[VEDARA] Warning: Config file not found at {self._config_path}")
            print("[VEDARA] Using default configuration values.")
            self._raw_config = {}
            return

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"[VEDARA] Error parsing config file: {e}")
            print("[VEDARA] Using default configuration values.")
            self._raw_config = {}

    def _get_nested(self, *keys: str, default: Any = None) -> Any:

        current = self._raw_config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _resolve_platform_value(self, section: Dict[str, Any], key: str, default: Any) -> Any:

        value = section.get(key, default)


        if isinstance(value, dict):
            if self._platform == PLATFORM_RPI and "rpi" in value:
                return value["rpi"]
            elif self._platform == PLATFORM_WINDOWS and "windows" in value:
                return value["windows"]
            elif "default" in value:
                return value["default"]

        return value

    def _parse_camera_config(self) -> CameraConfig:

        cam = self._raw_config.get("camera", {})


        if self._platform == PLATFORM_RPI:
            backend = cam.get("rpi", "picamera2")
        else:
            backend = cam.get("windows", "opencv:0")

        return CameraConfig(
            width=cam.get("width", 640),
            height=cam.get("height", 480),
            target_fps=cam.get("target_fps", 20),
            backend=backend,
            buffer_count=cam.get("buffer_count", 4),
            auto_exposure=cam.get("auto_exposure", True),
            auto_wb=cam.get("auto_wb", True)
        )

    def _parse_inference_config(self) -> InferenceConfig:

        inf = self._raw_config.get("inference", {})
        threads_config = inf.get("threads", {})


        if isinstance(threads_config, dict):
            if self._platform == PLATFORM_RPI:
                threads = threads_config.get("rpi", 4)
            else:
                threads = threads_config.get("windows", 4)
        else:
            threads = threads_config if isinstance(threads_config, int) else 4


        model_file = inf.get("model_file", "yolov5n_int8.tflite")
        model_path = self._project_root / "models" / model_file

        return InferenceConfig(
            model_file=model_file,
            model_path=model_path.resolve(),
            input_width=inf.get("input_width", 320),
            input_height=inf.get("input_height", 320),
            confidence_threshold=inf.get("confidence_threshold", 0.45),
            nms_threshold=inf.get("nms_threshold", 0.45),
            max_detections=inf.get("max_detections", 20),
            threads=threads
        )

    def _parse_hud_config(self) -> HUDConfig:

        hud = self._raw_config.get("hud", {})
        colors = hud.get("colors", {})
        glow = hud.get("glow", {})
        scanlines = hud.get("scanlines", {})
        animations = hud.get("animations", {})
        font = hud.get("font", {})


        default_colors = {
            "neon_cyan": [255, 243, 0],
            "electric_purple": [175, 42, 183],
            "hud_bg": [15, 10, 25],
            "hud_bg_alpha": 220,
            "text_primary": [255, 255, 255],
            "text_secondary": [180, 180, 180],
            "warning": [0, 128, 255],
            "danger": [0, 0, 255]
        }


        merged_colors = {**default_colors, **colors}

        glow_config = GlowConfig(
            enabled=glow.get("enabled", True),
            layers=glow.get("layers", 3),
            outer_thickness=glow.get("outer_thickness", 8),
            middle_thickness=glow.get("middle_thickness", 4),
            inner_thickness=glow.get("inner_thickness", 2),
            outer_alpha=glow.get("outer_alpha", 0.15),
            middle_alpha=glow.get("middle_alpha", 0.40),
            inner_alpha=glow.get("inner_alpha", 1.0)
        )

        return HUDConfig(
            enabled=hud.get("enabled", True),
            colors=merged_colors,
            glow=glow_config,
            scanlines_enabled=scanlines.get("enabled", True),
            scanlines_opacity=scanlines.get("opacity", 0.08),
            scanlines_spacing=scanlines.get("spacing", 4),
            animations_enabled=animations.get("enabled", True),
            slide_duration_ms=animations.get("slide_duration_ms", 150),
            fade_duration_ms=animations.get("fade_duration_ms", 100),
            font_scale=font.get("scale", 0.6),
            font_thickness=font.get("thickness", 1),
            outline_thickness=font.get("outline_thickness", 2)
        )

    def _parse_performance_config(self) -> PerformanceConfig:

        perf = self._raw_config.get("performance", {})
        thermal = perf.get("thermal", {})

        return PerformanceConfig(
            min_fps=perf.get("min_fps", 18),
            max_fps=perf.get("max_fps", 22),
            gc_interval_frames=perf.get("gc_interval_frames", 100),
            preallocate_buffers=perf.get("preallocate_buffers", True),
            thermal_enabled=thermal.get("enabled", True),
            throttle_temp_c=thermal.get("throttle_temp_c", 70),
            critical_temp_c=thermal.get("critical_temp_c", 80),
            thermal_check_interval=thermal.get("check_interval_frames", 50),
            max_cpu_percent=perf.get("max_cpu_percent", 85),
            adaptive_frameskip=perf.get("adaptive_frameskip", True)
        )

    def _parse_display_config(self) -> DisplayConfig:

        disp = self._raw_config.get("display", {})

        return DisplayConfig(
            title=disp.get("title", "VEDARA AR System"),
            fullscreen=disp.get("fullscreen", False),
            output_width=disp.get("output_width", 640),
            output_height=disp.get("output_height", 480),
            show_fps=disp.get("show_fps", True),
            show_cpu=disp.get("show_cpu", True),
            show_memory=disp.get("show_memory", True),
            show_temp=disp.get("show_temp", True)
        )

    def _parse_debug_config(self) -> DebugConfig:

        dbg = self._raw_config.get("debug", {})
        save_path = self._project_root / dbg.get("save_path", "debug/frames")

        return DebugConfig(
            simulate_rpi=dbg.get("simulate_rpi", False),
            throttle_cpu_ghz=dbg.get("throttle_cpu_ghz", 1.5),
            save_frames=dbg.get("save_frames", False),
            save_interval=dbg.get("save_interval", 100),
            save_path=save_path.resolve(),
            show_detection_time=dbg.get("show_detection_time", True),
            show_render_time=dbg.get("show_render_time", True)
        )

    @property
    def platform(self) -> PlatformType:

        return self._platform

    @property
    def project_root(self) -> Path:

        return self._project_root

    def reload(self) -> None:

        ConfigLoader._initialized = False
        self.__init__(self._config_path)

    def get_summary(self) -> Dict[str, Any]:

        return {
            "platform": self._platform,
            "config_path": str(self._config_path),
            "camera": {
                "resolution": f"{self.camera.width}x{self.camera.height}",
                "fps": self.camera.target_fps,
                "backend": self.camera.backend
            },
            "inference": {
                "model": self.inference.model_file,
                "input_size": f"{self.inference.input_width}x{self.inference.input_height}",
                "threads": self.inference.threads
            },
            "hud_enabled": self.hud.enabled,
            "performance": {
                "target_fps": f"{self.performance.min_fps}-{self.performance.max_fps}",
                "max_cpu": f"{self.performance.max_cpu_percent}%"
            }
        }


_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[Path] = None) -> ConfigLoader:

    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config() -> ConfigLoader:

    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()
    else:
        _config_instance = ConfigLoader()
    return _config_instance


if __name__ == "__main__":
    print("=" * 60)
    print("VEDARA Configuration Loader - Diagnostic Report")
    print("=" * 60)

    config = get_config()
    summary = config.get_summary()

    import json
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("Detailed Configuration:")
    print("=" * 60)

    print(f"\nCamera Config:")
    print(f"  Backend: {config.camera.backend}")
    print(f"  Resolution: {config.camera.width}x{config.camera.height}")
    print(f"  Target FPS: {config.camera.target_fps}")

    print(f"\nInference Config:")
    print(f"  Model Path: {config.inference.model_path}")
    print(f"  Threads: {config.inference.threads}")

    print(f"\nHUD Config:")
    print(f"  Enabled: {config.hud.enabled}")
    print(f"  Glow Enabled: {config.hud.glow.enabled}")
    print(f"  Colors: {list(config.hud.colors.keys())}")

    print(f"\nPerformance Config:")
    print(f"  FPS Range: {config.performance.min_fps}-{config.performance.max_fps}")
    print(f"  GC Interval: {config.performance.gc_interval_frames} frames")

    print("=" * 60)
