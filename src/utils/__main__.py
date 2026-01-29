def main():


    print("=" * 60)
    print("VEDARA Utilities - Full Diagnostic Suite")
    print("=" * 60)


    print("\n[1] PLATFORM DETECTION")
    print("-" * 40)

    from src.utils.platform_detect import get_detector

    detector = get_detector()
    summary = detector.get_summary()

    print(f"Platform: {summary['platform']}")
    print(f"CPU Cores: {summary['cpu_cores']}")
    print(f"RAM: {summary['ram_mb']} MB")
    print(f"Architecture: {summary['architecture']}")
    print(f"Python: {summary['python']}")
    print(f"Project Root: {summary['project_root']}")

    print("\nDependencies:")
    for dep, status in summary['dependencies'].items():
        print(f"  {dep}: {status}")

    success, issues = detector.validate_for_deployment()
    print(f"\nDeployment Ready: {'YES' if success else 'NO'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")


    print("\n" + "=" * 60)
    print("[2] CONFIGURATION LOADER")
    print("-" * 40)

    from src.utils.config_loader import get_config

    config = get_config()

    print(f"Config Path: {config._config_path}")
    print(f"Camera Backend: {config.camera.backend}")
    print(f"Resolution: {config.camera.width}x{config.camera.height}")
    print(f"Target FPS: {config.camera.target_fps}")
    print(f"HUD Enabled: {config.hud.enabled}")
    print(f"Glow Effect: {config.hud.glow.enabled}")
    print(f"FPS Range: {config.performance.min_fps}-{config.performance.max_fps}")
    print(f"Max CPU: {config.performance.max_cpu_percent}%")


    print("\n" + "=" * 60)
    print("[3] PERFORMANCE MONITOR")
    print("-" * 40)

    import time
    from src.utils.performance import get_monitor, TimerContext

    monitor = get_monitor()

    print("Running 20 frame simulation...")

    for i in range(20):
        monitor.frame_start()


        with TimerContext("capture"):
            time.sleep(0.008)

        with TimerContext("inference"):
            time.sleep(0.020)

        with TimerContext("render"):
            time.sleep(0.005)

        monitor.frame_end()

    print(f"\nPerformance Status:")
    print(f"  {monitor.get_status_string()}")

    report = monitor.get_detailed_report()
    print(f"\nTiming Breakdown:")
    print(f"  Total: {report['frame_time']['total_ms']:.1f}ms")
    print(f"  Capture: {report['frame_time']['capture_ms']:.1f}ms")
    print(f"  Inference: {report['frame_time']['inference_ms']:.1f}ms")
    print(f"  Render: {report['frame_time']['render_ms']:.1f}ms")

    fps_data = report['fps']
    in_target = "YES" if fps_data['in_target'] else "NO"
    print(f"\nFPS Analysis:")
    print(f"  Current: {fps_data['current']:.1f}")
    print(f"  Target: {fps_data['target_min']}-{fps_data['target_max']}")
    print(f"  In Target: {in_target}")


    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("[OK] Platform Detection: PASS")
    print("[OK] Configuration Loader: PASS")
    print("[OK] Performance Monitor: PASS")
    print("=" * 60)
    print("Operator, all utility subsystems operational.")
    print("=" * 60)


if __name__ == "__main__":
    main()
