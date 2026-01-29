import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.hud.effects import COLORS
from src.utils.config_loader import get_config


class FontStyle(Enum):

    NORMAL = cv2.FONT_HERSHEY_SIMPLEX
    BOLD = cv2.FONT_HERSHEY_DUPLEX
    SMALL = cv2.FONT_HERSHEY_PLAIN
    COMPLEX = cv2.FONT_HERSHEY_COMPLEX
    SCRIPT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    MONO = cv2.FONT_HERSHEY_COMPLEX_SMALL


@dataclass
class TextMetrics:

    width: int
    height: int
    baseline: int


class FontRenderer:


    def __init__(self) -> None:
        config = get_config()

        self.default_scale = config.hud.font_scale
        self.default_thickness = config.hud.font_thickness
        self.outline_thickness = config.hud.outline_thickness


        self._size_cache: Dict[str, TextMetrics] = {}
        self._cache_max_size = 100

    def measure_text(
        self,
        text: str,
        scale: float = None,
        thickness: int = None,
        font: FontStyle = FontStyle.NORMAL
    ) -> TextMetrics:

        if scale is None:
            scale = self.default_scale
        if thickness is None:
            thickness = self.default_thickness


        cache_key = f"{text}:{scale}:{thickness}:{font.value}"

        if cache_key in self._size_cache:
            return self._size_cache[cache_key]


        (width, height), baseline = cv2.getTextSize(
            text, font.value, scale, thickness
        )

        metrics = TextMetrics(width=width, height=height, baseline=baseline)


        if len(self._size_cache) < self._cache_max_size:
            self._size_cache[cache_key] = metrics

        return metrics

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        scale: float = None,
        thickness: int = None,
        font: FontStyle = FontStyle.NORMAL,
        outline: bool = True,
        outline_color: Tuple[int, int, int] = None,
        shadow: bool = False,
        shadow_offset: Tuple[int, int] = (2, 2),
        shadow_color: Tuple[int, int, int] = None
    ) -> np.ndarray:

        if color is None:
            color = COLORS.TEXT_PRIMARY
        if scale is None:
            scale = self.default_scale
        if thickness is None:
            thickness = self.default_thickness
        if outline_color is None:
            outline_color = (0, 0, 0)
        if shadow_color is None:
            shadow_color = (0, 0, 0)

        x, y = position
        font_face = font.value


        if shadow:
            sx, sy = shadow_offset
            cv2.putText(
                frame, text, (x + sx, y + sy),
                font_face, scale, shadow_color, thickness + 1
            )


        if outline:
            outline_thick = thickness + self.outline_thickness
            cv2.putText(
                frame, text, (x, y),
                font_face, scale, outline_color, outline_thick
            )


        cv2.putText(
            frame, text, (x, y),
            font_face, scale, color, thickness
        )

        return frame

    def draw_text_centered(
        self,
        frame: np.ndarray,
        text: str,
        center: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        scale: float = None,
        thickness: int = None,
        font: FontStyle = FontStyle.NORMAL,
        outline: bool = True
    ) -> np.ndarray:

        if scale is None:
            scale = self.default_scale
        if thickness is None:
            thickness = self.default_thickness

        metrics = self.measure_text(text, scale, thickness, font)

        x = center[0] - metrics.width // 2
        y = center[1] + metrics.height // 2

        return self.draw_text(
            frame, text, (x, y),
            color=color, scale=scale, thickness=thickness,
            font=font, outline=outline
        )

    def draw_text_boxed(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        bg_color: Tuple[int, int, int] = None,
        scale: float = None,
        thickness: int = None,
        padding: int = 5,
        font: FontStyle = FontStyle.NORMAL
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:

        if color is None:
            color = COLORS.TEXT_PRIMARY
        if bg_color is None:
            bg_color = COLORS.PANEL_BG
        if scale is None:
            scale = self.default_scale
        if thickness is None:
            thickness = self.default_thickness

        metrics = self.measure_text(text, scale, thickness, font)

        x, y = position
        box_x = x - padding
        box_y = y - metrics.height - padding
        box_w = metrics.width + padding * 2
        box_h = metrics.height + padding * 2 + metrics.baseline


        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            bg_color, -1
        )


        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            color, 1
        )


        cv2.putText(
            frame, text, (x, y),
            font.value, scale, color, thickness
        )

        return frame, (box_x, box_y, box_w, box_h)

    def draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        bg_alpha: float = 0.7
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN

        scale = self.default_scale * 0.8
        thickness = 1
        padding = 3

        metrics = self.measure_text(text, scale, thickness)

        x, y = position
        box_x1 = x
        box_y1 = y - metrics.height - padding * 2
        box_x2 = x + metrics.width + padding * 2
        box_y2 = y


        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), COLORS.PANEL_BG, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)


        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 1)


        cv2.putText(
            frame, text, (x + padding, y - padding),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
        )

        return frame

    def draw_multiline(
        self,
        frame: np.ndarray,
        lines: list,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        scale: float = None,
        line_spacing: float = 1.5,
        font: FontStyle = FontStyle.NORMAL,
        outline: bool = True
    ) -> np.ndarray:

        if color is None:
            color = COLORS.TEXT_PRIMARY
        if scale is None:
            scale = self.default_scale

        x, y = position

        for line in lines:
            metrics = self.measure_text(line, scale, self.default_thickness, font)

            self.draw_text(
                frame, line, (x, y),
                color=color, scale=scale, font=font, outline=outline
            )

            y += int(metrics.height * line_spacing)

        return frame


class StatusText:


    def __init__(self) -> None:
        self.font = FontRenderer()

    def draw(
        self,
        frame: np.ndarray,
        label: str,
        value: str,
        position: Tuple[int, int],
        status: str = "normal"
    ) -> np.ndarray:


        status_colors = {
            "ok": COLORS.STATUS_OK,
            "warn": COLORS.STATUS_WARN,
            "danger": COLORS.STATUS_DANGER,
            "normal": COLORS.TEXT_PRIMARY
        }
        value_color = status_colors.get(status, COLORS.TEXT_PRIMARY)

        x, y = position


        self.font.draw_text(
            frame, label, (x, y),
            color=COLORS.TEXT_SECONDARY,
            scale=0.5, outline=True
        )


        label_metrics = self.font.measure_text(label, 0.5)
        self.font.draw_text(
            frame, value, (x + label_metrics.width + 5, y),
            color=value_color,
            scale=0.5, outline=True
        )

        return frame


class TitleText:


    def __init__(self) -> None:
        self.font = FontRenderer()

    def draw(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN

        return self.font.draw_text(
            frame, text, position,
            color=color,
            scale=0.9,
            thickness=2,
            font=FontStyle.BOLD,
            outline=True,
            shadow=True
        )


_font_renderer: Optional[FontRenderer] = None
_status_text: Optional[StatusText] = None
_title_text: Optional[TitleText] = None


def get_font_renderer() -> FontRenderer:

    global _font_renderer
    if _font_renderer is None:
        _font_renderer = FontRenderer()
    return _font_renderer


def get_status_text() -> StatusText:

    global _status_text
    if _status_text is None:
        _status_text = StatusText()
    return _status_text


def get_title_text() -> TitleText:

    global _title_text
    if _title_text is None:
        _title_text = TitleText()
    return _title_text
