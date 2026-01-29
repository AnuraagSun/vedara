import numpy as np
import cv2
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from src.hud.effects import (
    COLORS,
    get_glow_effect,
    get_scanline_effect,
    get_corner_brackets,
    get_targeting_reticle,
    get_data_panel,
    get_progress_bar
)
from src.hud.fonts import (
    get_font_renderer,
    get_status_text,
    get_title_text,
    FontStyle
)
from src.hud.animations import (
    get_animator,
    get_detection_card_animator,
    get_pulse
)
from src.inference.detector import Detection, InferenceResult
from src.utils.config_loader import get_config
from src.utils.performance import get_monitor, PerformanceMetrics
from src.utils.platform_detect import get_platform, PLATFORM_RPI


@dataclass
class HUDLayout:


    width: int = 640
    height: int = 480


    status_x: int = 10
    status_y: int = 25
    status_line_height: int = 20


    title_y: int = 25


    card_offset_x: int = 10
    card_width: int = 150
    card_padding: int = 8


    perf_x: int = 10
    perf_y_offset: int = 20

    def update_dimensions(self, width: int, height: int) -> None:

        self.width = width
        self.height = height
        self.perf_y_offset = height - 20


class HUDRenderer:


    def __init__(self) -> None:
        config = get_config()
        self._enabled = config.hud.enabled


        self._layout = HUDLayout()


        self._glow = get_glow_effect()
        self._scanlines = get_scanline_effect()
        self._brackets = get_corner_brackets()
        self._reticle = get_targeting_reticle()
        self._panel = get_data_panel()
        self._progress = get_progress_bar()


        self._font = get_font_renderer()
        self._status_text = get_status_text()
        self._title = get_title_text()


        self._animator = get_animator()
        self._card_animator = get_detection_card_animator()
        self._pulse = get_pulse()


        self._perf_monitor = get_monitor()
        self._platform = get_platform()


        self._frame_count = 0
        self._show_fps = config.display.show_fps
        self._show_cpu = config.display.show_cpu
        self._show_memory = config.display.show_memory
        self._show_temp = config.display.show_temp and self._platform == PLATFORM_RPI


        self._overlay_buffer: Optional[np.ndarray] = None

    def render(
        self,
        frame: np.ndarray,
        detections: Optional[InferenceResult] = None,
        show_performance: bool = True,
        show_title: bool = True
    ) -> np.ndarray:

        if not self._enabled:
            return frame

        self._frame_count += 1
        height, width = frame.shape[:2]
        self._layout.update_dimensions(width, height)


        if detections and detections.detections:
            self._render_detections(frame, detections.detections)


        if show_title:
            self._render_title(frame)


        if show_performance:
            self._render_performance(frame)


        self._scanlines.apply(frame)

        return frame

    def _render_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> None:

        active_ids = set()

        for i, det in enumerate(detections):
            detection_id = det.track_id if det.track_id is not None else i
            active_ids.add(detection_id)

            bbox = det.bbox
            pt1 = (bbox.x, bbox.y)
            pt2 = (bbox.x + bbox.w, bbox.y + bbox.h)
            center = bbox.center


            if det.confidence >= 0.8:
                color = COLORS.NEON_CYAN
            elif det.confidence >= 0.6:
                color = COLORS.ELECTRIC_PURPLE
            else:
                color = COLORS.STATUS_WARN


            self._glow.draw_glowing_box(frame, pt1, pt2, color)


            self._brackets.draw(frame, pt1, pt2, color)


            if i == 0:
                reticle_size = min(bbox.w, bbox.h) // 4
                self._reticle.draw(frame, center, size=max(15, reticle_size), color=color)


            label = f"{det.class_name} {det.confidence:.0%}"
            self._font.draw_label(frame, label, pt1, color=color)


            self._render_detection_card(frame, det, detection_id, pt2)


        self._card_animator.cleanup_old_cards(active_ids)

    def _render_detection_card(
        self,
        frame: np.ndarray,
        detection: Detection,
        detection_id: int,
        anchor_pt: Tuple[int, int]
    ) -> None:

        layout = self._layout
        bbox = detection.bbox


        target_x = anchor_pt[0] + layout.card_offset_x
        card_y = bbox.y


        self._card_animator.on_detection_appear(detection_id, target_x)


        card_x = self._card_animator.get_card_position(detection_id, target_x)
        card_alpha = self._card_animator.get_card_alpha(detection_id)


        if card_x + layout.card_width > layout.width:

            card_x = bbox.x - layout.card_width - layout.card_offset_x

        if card_x < 0:
            return


        card_height = 70
        card_pt1 = (card_x, card_y)
        card_pt2 = (card_x + layout.card_width, card_y + card_height)


        self._panel.draw(frame, card_pt1, card_pt2, alpha=0.75 * card_alpha)


        padding = layout.card_padding
        text_x = card_x + padding
        text_y = card_y + padding + 12


        self._font.draw_text(
            frame, detection.class_name.upper(),
            (text_x, text_y),
            color=COLORS.NEON_CYAN,
            scale=0.45,
            outline=True
        )


        bar_y = text_y + 8
        self._progress.draw(
            frame,
            (text_x, bar_y),
            width=layout.card_width - padding * 2,
            height=8,
            progress=detection.confidence,
            color=COLORS.NEON_CYAN
        )


        conf_text = f"CONF: {detection.confidence:.0%}"
        self._font.draw_text(
            frame, conf_text,
            (text_x, bar_y + 22),
            color=COLORS.TEXT_SECONDARY,
            scale=0.35,
            outline=True
        )


        id_text = f"ID:{detection_id:03d}"
        self._font.draw_text(
            frame, id_text,
            (text_x, bar_y + 38),
            color=COLORS.TEXT_SECONDARY,
            scale=0.3,
            outline=True
        )

    def _render_title(self, frame: np.ndarray) -> None:

        title = "VEDARA"
        subtitle = "AR VISION SYSTEM"


        alpha = self._pulse.get_alpha()


        center_x = self._layout.width // 2
        self._font.draw_text_centered(
            frame, title,
            (center_x, 25),
            color=COLORS.NEON_CYAN,
            scale=0.7,
            thickness=2
        )


        self._font.draw_text_centered(
            frame, subtitle,
            (center_x, 45),
            color=COLORS.TEXT_SECONDARY,
            scale=0.4,
            thickness=1
        )


        dot_x = center_x + 60
        pulse_radius = int(4 + 2 * (alpha - 0.5))
        cv2.circle(frame, (dot_x, 20), pulse_radius, COLORS.STATUS_OK, -1)

    def _render_performance(self, frame: np.ndarray) -> None:

        metrics = self._perf_monitor.get_metrics()
        layout = self._layout

        x = layout.status_x
        y = layout.height - 60
        line_height = 16


        panel_width = 140
        panel_height = 55 if not self._show_temp else 70
        self._panel.draw(
            frame,
            (x - 5, y - 5),
            (x + panel_width, y + panel_height),
            alpha=0.6
        )


        if self._show_fps:
            fps_status = "ok" if 18 <= metrics.fps <= 25 else "warn" if metrics.fps >= 15 else "danger"
            self._status_text.draw(
                frame, "FPS:", f"{metrics.fps:.1f}",
                (x, y), status=fps_status
            )
            y += line_height


        if self._show_cpu:
            cpu_status = "ok" if metrics.cpu_percent < 70 else "warn" if metrics.cpu_percent < 85 else "danger"
            self._status_text.draw(
                frame, "CPU:", f"{metrics.cpu_percent:.0f}%",
                (x, y), status=cpu_status
            )
            y += line_height


        if self._show_memory:
            mem_status = "ok" if metrics.memory_used_mb < 2500 else "warn" if metrics.memory_used_mb < 3000 else "danger"
            self._status_text.draw(
                frame, "RAM:", f"{metrics.memory_used_mb:.0f}MB",
                (x, y), status=mem_status
            )
            y += line_height


        if self._show_temp and metrics.temperature_c > 0:
            temp_status = "ok" if metrics.temperature_c < 60 else "warn" if metrics.temperature_c < 75 else "danger"
            self._status_text.draw(
                frame, "TEMP:", f"{metrics.temperature_c:.0f}°C",
                (x, y), status=temp_status
            )

            if metrics.thermal_throttled:
                self._font.draw_text(
                    frame, "THROTTLED",
                    (x, y + line_height),
                    color=COLORS.STATUS_DANGER,
                    scale=0.35
                )

    def set_enabled(self, enabled: bool) -> None:

        self._enabled = enabled

    def is_enabled(self) -> bool:

        return self._enabled

    def get_frame_count(self) -> int:

        return self._frame_count


_hud_renderer: Optional[HUDRenderer] = None


def get_hud_renderer() -> HUDRenderer:

    global _hud_renderer
    if _hud_renderer is None:
        _hud_renderer = HUDRenderer()
    return _hud_renderer


def render_hud(
    frame: np.ndarray,
    detections: Optional[InferenceResult] = None,
    show_performance: bool = True,
    show_title: bool = True
) -> np.ndarray:

    return get_hud_renderer().render(frame, detections, show_performance, show_title)
