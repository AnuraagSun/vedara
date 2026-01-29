def main():


    import json
    import numpy as np

    print("=" * 60)
    print("VEDARA Inference Engine - Diagnostic Test")
    print("=" * 60)

    from src.inference import (
        DetectorFactory,
        create_detector,
        TFLITE_AVAILABLE,
        TFLITE_SOURCE
    )


    print("\n[1] Available Backends")
    print("-" * 40)

    backends = DetectorFactory.get_available_backends()
    for name, available in backends.items():
        status = "AVAILABLE" if available else "NOT AVAILABLE"
        symbol = "[OK]" if available else "[--]"
        print(f"  {symbol} {name}: {status}")

    if TFLITE_AVAILABLE:
        print(f"\n  TFLite Source: {TFLITE_SOURCE}")
    else:
        print("\n  Note: Install tflite-runtime for real inference")
        print("        pip install tflite-runtime")


    print("\n" + "=" * 60)
    print("[2] Detector Initialization")
    print("-" * 40)

    detector = create_detector(use_mock_if_unavailable=True)

    print(f"Detector Type: {type(detector).__name__}")
    print(f"Initialized: {detector.is_initialized}")
    print(f"Model Name: {detector.model_name}")
    print(f"Input Size: {detector.input_size[0]}x{detector.input_size[1]}")


    print("\n" + "=" * 60)
    print("[3] Test Inference")
    print("-" * 40)


    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        dummy_frame[y, :, 0] = int(25 + (y / 480) * 30)
        dummy_frame[y, :, 1] = 10
        dummy_frame[y, :, 2] = int(15 + (y / 480) * 20)

    print("Running 5 inference cycles...")
    print()

    total_time = 0.0
    for i in range(5):
        result = detector.detect(dummy_frame)
        total_time += result.total_time_ms

        print(f"Frame {result.frame_number}:")
        print(f"  Timing: {result.total_time_ms:.1f}ms "
              f"(pre:{result.preprocess_time_ms:.1f} + "
              f"inf:{result.inference_time_ms:.1f} + "
              f"post:{result.postprocess_time_ms:.1f})")
        print(f"  Detections: {result.detection_count}")

        for det in result.detections:
            bbox = det.bbox
            print(f"    [{det.class_id:2d}] {det.class_name:15s} "
                  f"conf:{det.confidence:.2f} "
                  f"box:({bbox.x},{bbox.y},{bbox.w}x{bbox.h})")
        print()

    avg_time = total_time / 5
    estimated_fps = 1000.0 / avg_time if avg_time > 0 else 0

    print("-" * 40)
    print(f"Average Time: {avg_time:.1f}ms")
    print(f"Estimated FPS: {estimated_fps:.1f}")


    print("\n" + "=" * 60)
    print("[4] Diagnostics")
    print("-" * 40)

    diag = detector.get_diagnostics()
    print(json.dumps(diag, indent=2))


    detector.release()

    print("\n" + "=" * 60)
    print("INFERENCE ENGINE TEST COMPLETE")
    print("=" * 60)
    print("Operator, neural subsystem operational.")
    if not TFLITE_AVAILABLE:
        print("Note: Using mock detector. Install tflite-runtime for real models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
