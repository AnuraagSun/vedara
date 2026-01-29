import time
import json


def main() -> None:


    from src.camera import (
        CameraFactory,
        create_camera,
        OPENCV_AVAILABLE,
        PICAMERA2_AVAILABLE
    )

    print("=" * 60)
    print("VEDARA Camera Factory - Integration Test")
    print("=" * 60)


    print("\nAvailable Backends:")
    backends = CameraFactory.get_available_backends()
    for name, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {name}: {status}")


    recommended = CameraFactory.get_recommended_backend()
    print(f"\nRecommended backend: {recommended}")


    print("\n" + "-" * 60)
    print("Creating camera with auto-detection...")

    try:
        camera = create_camera()
        print(f"Camera type: {type(camera).__name__}")
        print(f"Initial state: {camera.state.name}")


        print("\nInitializing camera...")
        success = camera.initialize(width=640, height=480, fps=30)

        if success:
            print(f"Initialization: SUCCESS")
            print(f"Properties: {camera.properties.width}x{camera.properties.height} @ {camera.properties.fps}fps")
            print(f"Backend: {camera.properties.backend_name}")


            camera.start_streaming()


            print("\nCapturing 5 test frames...")
            for i in range(5):
                frame_data = camera.capture_frame()
                if frame_data:
                    print(f"  Frame {frame_data.frame_number}: "
                          f"{frame_data.width}x{frame_data.height} - "
                          f"Valid: {frame_data.is_valid}")
                time.sleep(0.1)


            print("\nCamera Diagnostics:")
            diag = camera.get_diagnostics()
            print(json.dumps(diag, indent=2, default=str))


            camera.release()
            print(f"\nFinal state: {camera.state.name}")
        else:
            print(f"Initialization FAILED: {camera.last_error}")


            print("\n" + "-" * 60)
            print("Testing NullCamera fallback...")

            from src.camera import NullCamera
            null_cam = NullCamera()
            null_cam.initialize(640, 480, 30)
            null_cam.start_streaming()

            for i in range(3):
                frame_data = null_cam.capture_frame()
                if frame_data:
                    print(f"  NullCamera Frame {frame_data.frame_number}: "
                          f"{frame_data.width}x{frame_data.height}")

            null_cam.release()
            print("NullCamera test: SUCCESS")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Camera factory test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
