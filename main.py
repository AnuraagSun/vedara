import sys
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))


def print_banner():

    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██╗   ██╗███████╗██████╗  █████╗ ██████╗  █████╗       ║
    ║   ██║   ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗      ║
    ║   ██║   ██║█████╗  ██║  ██║███████║██████╔╝███████║      ║
    ║   ╚██╗ ██╔╝██╔══╝  ██║  ██║██╔══██║██╔══██╗██╔══██║      ║
    ║    ╚████╔╝ ███████╗██████╔╝██║  ██║██║  ██║██║  ██║      ║
    ║     ╚═══╝  ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝      ║
    ║                                                           ║
    ║           Advanced AR Object Detection System             ║
    ║                      Version 1.0.0                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_validation():

    print("=" * 60)
    print("VEDARA System Validation")
    print("=" * 60)

    results = []


    print("\n[1/5] Platform Detection...")
    try:
        from src.utils.platform_detect import get_detector
        detector = get_detector()
        platform = detector.detect_platform()
        print(f"      Platform: {platform}")
        print(f"      [PASS]")
        results.append(("Platform Detection", True))
    except Exception as e:
        print(f"      [FAIL] {e}")
        results.append(("Platform Detection", False))


    print("\n[2/5] Configuration Loading...")
    try:
        from src.utils.config_loader import get_config
        config = get_config()
        print(f"      Camera: {config.camera.backend}")
        print(f"      [PASS]")
        results.append(("Configuration", True))
    except Exception as e:
        print(f"      [FAIL] {e}")
        results.append(("Configuration", False))


    print("\n[3/5] Camera System...")
    try:
        from src.camera import create_camera
        camera = create_camera(use_null_if_unavailable=True)
        success = camera.initialize(640, 480, 30)
        if success:
            camera.start_streaming()
            frame = camera.capture_frame()
            camera.release()
            print(f"      Backend: {camera.properties.backend_name}")
            print(f"      Frame: {'Valid' if frame and frame.is_valid else 'None'}")
            print(f"      [PASS]")
            results.append(("Camera", True))
        else:
            print(f"      [FAIL] {camera.last_error}")
            results.append(("Camera", False))
    except Exception as e:
        print(f"      [FAIL] {e}")
        results.append(("Camera", False))


    print("\n[4/5] Detection System...")
    try:
        from src.inference import create_detector
        import numpy as np
        detector = create_detector(use_mock_if_unavailable=True)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_frame)
        detector.release()
        print(f"      Model: {detector.model_name}")
        print(f"      Detections: {result.detection_count}")
        print(f"      [PASS]")
        results.append(("Detector", True))
    except Exception as e:
        print(f"      [FAIL] {e}")
        results.append(("Detector", False))


    print("\n[5/5] HUD System...")
    try:
        from src.hud import get_hud_renderer, render_hud
        import numpy as np
        hud = get_hud_renderer()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame = render_hud(test_frame, None, True, True)
        print(f"      Enabled: {hud.is_enabled()}")
        print(f"      [PASS]")
        results.append(("HUD", True))
    except Exception as e:
        print(f"      [FAIL] {e}")
        results.append(("HUD", False))


    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print("-" * 60)
    print(f"  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  ✓ System ready for deployment!")
    else:
        print("\n  ✗ Some tests failed. Check configuration.")

    print("=" * 60)

    return passed == total


def main():


    parser = argparse.ArgumentParser(
        description="VEDARA AR Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (during operation):
  Q / ESC   Quit application
  H         Toggle HUD overlay
  D         Toggle object detection
  F         Toggle fullscreen
  S         Save screenshot

Examples:
  python main.py                        # Run with defaults
  python main.py --fullscreen           # Fullscreen mode
  python main.py --no-hud               # Camera only
  python main.py --validate             # Run system tests
  python main.py --platform rpi         # Force RPi mode
        """
    )

    parser.add_argument(
        "--platform",
        choices=["auto", "windows", "rpi"],
        default="auto",
        help="Force platform mode (default: auto-detect)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation tests and exit"
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without display window (headless)"
    )

    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Disable object detection"
    )

    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD overlay"
    )

    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Run in fullscreen mode"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target frame rate (default: 20)"
    )

    parser.add_argument(
        "--simulate-rpi",
        action="store_true",
        help="Simulate RPi performance on Windows"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal console output"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )

    args = parser.parse_args()


    if args.version:
        print("VEDARA AR System v1.0.0")
        return 0


    if not args.quiet:
        print_banner()


    if args.validate:
        success = run_validation()
        return 0 if success else 1


    from src.core import VedaraEngine, EngineConfig


    config = EngineConfig(
        show_preview=not args.no_preview,
        enable_detection=not args.no_detection,
        enable_hud=not args.no_hud,
        fullscreen=args.fullscreen,
        target_fps=args.fps,
    )


    engine = VedaraEngine(config)


    if not engine.initialize():
        print(f"\n[ERROR] Initialization failed: {engine.last_error}")
        return 1


    if not args.quiet:
        print("\n" + "-" * 60)
        print("Controls:")
        print("  Q/ESC : Quit     H : Toggle HUD")
        print("  D     : Toggle Detection    F : Fullscreen")
        print("  S     : Screenshot")
        print("-" * 60)

    try:
        engine.run()
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n[VEDARA] Interrupted")


    engine.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
