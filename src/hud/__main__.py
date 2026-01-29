import numpy as np
import cv2
import time


def main():


    print("=" * 60)
    print("VEDARA HUD System - Comprehensive Visual Test")
    print("=" * 60)


    from src.hud import (
        COLORS,
        get_glow_effect,
        get_scanline_effect,
        get_corner_brackets,
        get_targeting_reticle,
        get_data_panel,
        get_progress_bar,
        get_font_renderer,
        get_hud_renderer,
        render_hud,
    )
    from src.inference import (
        Detection,
        BoundingBox,
        InferenceResult,
    )


    print("\nInitializing HUD subsystems...")

    glow = get_glow_effect()
    scanlines = get_scanline_effect()
    brackets = get_corner_brackets()
    reticle = get_targeting_reticle()
    panel = get_data_panel()
    progress = get_progress_bar()
    font = get_font_renderer()
    hud = get_hud_renderer()

    print(f"  Glow Effect: {'ON' if glow.enabled else 'OFF'} ({glow.layers} layers)")
    print(f"  Scanlines: {'ON' if scanlines.enabled else 'OFF'}")
    print(f"  HUD Renderer: {'Enabled' if hud.is_enabled() else 'Disabled'}")


    print("\nCreating mock detections...")

    def create_mock_detections(frame_num: int) -> InferenceResult:

        detections = []


        t = (frame_num % 100) / 100.0
        x_offset = 0.1 + 0.3 * abs(np.sin(t * np.pi * 2))

        bbox1 = BoundingBox(
            x_norm=x_offset,
            y_norm=0.15,
            w_norm=0.18,
            h_norm=0.5
        ).to_pixel_coords(640, 480)

        detections.append(Detection(
            class_id=0,
            class_name="person",
            confidence=0.94,
            bbox=bbox1,
            track_id=1
        ))


        bbox2 = BoundingBox(
            x_norm=0.6,
            y_norm=0.4,
            w_norm=0.1,
            h_norm=0.18
        ).to_pixel_coords(640, 480)

        detections.append(Detection(
            class_id=67,
            class_name="cell phone",
            confidence=0.82,
            bbox=bbox2,
            track_id=2
        ))


        if (frame_num % 60) < 40:
            bbox3 = BoundingBox(
                x_norm=0.35,
                y_norm=0.55,
                w_norm=0.12,
                h_norm=0.15
            ).to_pixel_coords(640, 480)

            detections.append(Detection(
                class_id=41,
                class_name="cup",
                confidence=0.71,
                bbox=bbox3,
                track_id=3
            ))

        return InferenceResult(
            detections=detections,
            inference_time_ms=18.5,
            preprocess_time_ms=3.2,
            postprocess_time_ms=2.1,
            frame_number=frame_num,
            timestamp=time.time(),
            model_name="mock_detector"
        )


    print("\nStarting render test...")
    print("Press 'Q' to quit, 'S' to save screenshot, 'G' to toggle glow")
    print("-" * 60)

    frame_count = 0
    start_time = time.perf_counter()
    running = True
    save_mode = False

    try:

        cv2.namedWindow("VEDARA HUD Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VEDARA HUD Test", 960, 720)
    except Exception as e:
        print(f"[!] Cannot create display window: {e}")
        print("[!] Running in headless mode - will save frames")
        save_mode = True
        running = False

    while running:
        frame_count += 1


        frame = np.zeros((480, 640, 3), dtype=np.uint8)


        for y in range(480):
            intensity = int(15 + (y / 480) * 20)
            frame[y, :] = [intensity + 10, intensity, intensity + 15]


        noise = np.random.randint(0, 10, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)


        detections = create_mock_detections(frame_count)


        frame = render_hud(
            frame,
            detections=detections,
            show_performance=True,
            show_title=True
        )


        try:
            cv2.imshow("VEDARA HUD Test", frame)

            key = cv2.waitKey(33) & 0xFF

            if key == ord('q') or key == 27:
                running = False
            elif key == ord('s'):
                filename = f"vedara_hud_frame_{frame_count:04d}.png"
                cv2.imwrite(filename, frame)
                print(f"[OK] Saved: {filename}")
            elif key == ord('g'):
                glow.enabled = not glow.enabled
                print(f"[OK] Glow: {'ON' if glow.enabled else 'OFF'}")

        except Exception:
            running = False


        if frame_count % 30 == 0:
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed
            print(f"  Frame {frame_count}: {fps:.1f} FPS, "
                  f"Detections: {detections.detection_count}")


    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


    if save_mode or frame_count < 10:
        print("\nGenerating static test image...")


        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            intensity = int(15 + (y / 480) * 20)
            frame[y, :] = [intensity + 10, intensity, intensity + 15]

        detections = create_mock_detections(25)
        frame = render_hud(frame, detections, True, True)

        filename = "vedara_hud_test.png"
        cv2.imwrite(filename, frame)
        print(f"[OK] Saved: {filename}")
        print(f"[OK] Open with: start {filename}")


    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("HUD SYSTEM TEST COMPLETE")
    print("=" * 60)
    print(f"Frames Rendered: {frame_count}")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"HUD Frame Count: {hud.get_frame_count()}")
    print("=" * 60)
    print("Operator, cyberpunk HUD system fully operational.")
    print("Visual aesthetics: CONFIRMED")
    print("=" * 60)


if __name__ == "__main__":
    main()
