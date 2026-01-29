import argparse
import sys


def main():


    parser = argparse.ArgumentParser(
        description="VEDARA AR Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.core                    # Run with defaults
  python -m src.core --no-preview       # Run headless
  python -m src.core --fullscreen       # Fullscreen mode
  python -m src.core --no-detection     # HUD only (no inference)
        """
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (headless mode)"
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
        "--target-fps",
        type=int,
        default=20,
        help="Target frame rate (default: 20)"
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames to output directory"
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print diagnostics and exit"
    )

    args = parser.parse_args()


    print("=" * 60)
    print("VEDARA AR Object Detection System")
    print("=" * 60)

    from src.core import VedaraEngine, EngineConfig
    from src.utils.platform_detect import get_detector


    if args.diagnostics:
        print("\n[DIAGNOSTICS MODE]")

        detector = get_detector()
        print("\nPlatform Information:")
        for key, value in detector.get_summary().items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        success, issues = detector.validate_for_deployment()
        print(f"\nDeployment Ready: {'YES' if success else 'NO'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

        return 0


    config = EngineConfig(
        show_preview=not args.no_preview,
        enable_detection=not args.no_detection,
        enable_hud=not args.no_hud,
        fullscreen=args.fullscreen,
        target_fps=args.target_fps,
        save_frames=args.save_frames,
    )


    engine = VedaraEngine(config)


    if not engine.initialize():
        print(f"\n[ERROR] Initialization failed: {engine.last_error}")
        return 1


    print("\n[VEDARA] Starting AR system...")
    print("[VEDARA] Controls:")
    print("  Q/ESC : Quit")
    print("  H     : Toggle HUD")
    print("  D     : Toggle Detection")
    print("  F     : Toggle Fullscreen")
    print("  S     : Save Screenshot")
    print("-" * 60)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n[VEDARA] Interrupted by user")


    engine.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
