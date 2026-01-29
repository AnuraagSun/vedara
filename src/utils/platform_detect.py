import platform
import os
import sys
from pathlib import Path
from typing import Literal, Dict, Any
from dataclasses import dataclass, field


PLATFORM_WINDOWS = "windows"
PLATFORM_RPI = "rpi"
PLATFORM_LINUX = "linux"
PLATFORM_UNKNOWN = "unknown"

PlatformType = Literal["windows", "rpi", "linux", "unknown"]


@dataclass
class SystemCapabilities:

    platform: PlatformType
    cpu_count: int
    total_ram_mb: int
    is_arm: bool
    is_64bit: bool
    python_version: str
    has_picamera: bool = False
    has_opencv: bool = False
    has_tflite: bool = False
    project_root: Path = field(default_factory=Path)

    def __post_init__(self) -> None:

        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()


class PlatformDetector:


    def __init__(self) -> None:
        self._capabilities: SystemCapabilities | None = None
        self._project_root: Path | None = None

    @property
    def project_root(self) -> Path:

        if self._project_root is None:


            current_file = Path(__file__).resolve()
            self._project_root = current_file.parent.parent.parent
        return self._project_root

    def detect_platform(self) -> PlatformType:

        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "windows":
            return PLATFORM_WINDOWS

        if system == "linux":

            if self._is_raspberry_pi():
                return PLATFORM_RPI
            return PLATFORM_LINUX

        return PLATFORM_UNKNOWN

    def _is_raspberry_pi(self) -> bool:


        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            try:
                content = cpuinfo_path.read_text().lower()
                if "raspberry pi" in content or "bcm2" in content:
                    return True
            except (PermissionError, IOError):
                pass


        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            try:
                content = model_path.read_text().lower()
                if "raspberry pi" in content:
                    return True
            except (PermissionError, IOError):
                pass


        rpi_markers = [
            Path("/sys/firmware/devicetree/base/model"),
            Path("/opt/vc/lib"),
        ]
        for marker in rpi_markers:
            if marker.exists():
                return True

        return False

    def _check_dependency(self, module_name: str) -> bool:

        import importlib.util
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def _get_ram_mb(self) -> int:

        try:
            import psutil
            return int(psutil.virtual_memory().total / (1024 * 1024))
        except ImportError:

            return 4096 if self.detect_platform() == PLATFORM_RPI else 8192

    def get_capabilities(self) -> SystemCapabilities:

        if self._capabilities is not None:
            return self._capabilities

        detected_platform = self.detect_platform()
        machine = platform.machine().lower()


        is_arm = machine in ("aarch64", "armv7l", "armv8l", "arm64")
        is_64bit = sys.maxsize > 2**32


        has_picamera = self._check_dependency("picamera2")
        has_opencv = self._check_dependency("cv2")
        has_tflite = (
            self._check_dependency("tflite_runtime") or
            self._check_dependency("tensorflow")
        )

        self._capabilities = SystemCapabilities(
            platform=detected_platform,
            cpu_count=os.cpu_count() or 4,
            total_ram_mb=self._get_ram_mb(),
            is_arm=is_arm,
            is_64bit=is_64bit,
            python_version=platform.python_version(),
            has_picamera=has_picamera,
            has_opencv=has_opencv,
            has_tflite=has_tflite,
            project_root=self.project_root
        )

        return self._capabilities

    def get_summary(self) -> Dict[str, Any]:

        caps = self.get_capabilities()
        return {
            "platform": caps.platform,
            "cpu_cores": caps.cpu_count,
            "ram_mb": caps.total_ram_mb,
            "architecture": "ARM64" if caps.is_arm else "x86_64",
            "python": caps.python_version,
            "dependencies": {
                "picamera2": "✓" if caps.has_picamera else "✗",
                "opencv": "✓" if caps.has_opencv else "✗",
                "tflite": "✓" if caps.has_tflite else "✗",
            },
            "project_root": str(caps.project_root),
        }

    def validate_for_deployment(self) -> tuple[bool, list[str]]:

        caps = self.get_capabilities()
        issues: list[str] = []


        if caps.total_ram_mb < 2048:
            issues.append(f"Insufficient RAM: {caps.total_ram_mb}MB (minimum 2048MB)")


        if not caps.has_opencv:
            issues.append("OpenCV not installed. Run: pip install opencv-python-headless")

        if not caps.has_tflite:
            issues.append("TFLite runtime not installed. Run: pip install tflite-runtime")


        if caps.platform == PLATFORM_RPI and not caps.has_picamera:
            issues.append("Picamera2 not installed on RPi. Run: pip install picamera2")

        return (len(issues) == 0, issues)


_detector_instance: PlatformDetector | None = None


def get_detector() -> PlatformDetector:

    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlatformDetector()
    return _detector_instance


def get_platform() -> PlatformType:

    return get_detector().detect_platform()


def get_project_root() -> Path:

    return get_detector().project_root


if __name__ == "__main__":
    print("=" * 60)
    print("VEDARA Platform Detection - Diagnostic Report")
    print("=" * 60)

    detector = get_detector()
    summary = detector.get_summary()

    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("Deployment Validation:")
    print("=" * 60)

    success, issues = detector.validate_for_deployment()
    if success:
        print("✓ All systems nominal. Ready for deployment.")
    else:
        print("✗ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 60)
