import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import cv2

from src.utils.config_loader import get_config


@dataclass(frozen=True)
class ColorPalette:


    NEON_CYAN: Tuple[int, int, int] = (255, 243, 0)
    ELECTRIC_PURPLE: Tuple[int, int, int] = (175, 42, 183)
    HOT_PINK: Tuple[int, int, int] = (147, 20, 255)


    DARK_BG: Tuple[int, int, int] = (15, 10, 25)
    PANEL_BG: Tuple[int, int, int] = (25, 15, 35)


    TEXT_PRIMARY: Tuple[int, int, int] = (255, 255, 255)
    TEXT_SECONDARY: Tuple[int, int, int] = (180, 180, 180)
    TEXT_ACCENT: Tuple[int, int, int] = (255, 243, 0)


    STATUS_OK: Tuple[int, int, int] = (0, 255, 128)
    STATUS_WARN: Tuple[int, int, int] = (0, 165, 255)
    STATUS_DANGER: Tuple[int, int, int] = (0, 0, 255)


    ALPHA_FULL: int = 255
    ALPHA_HIGH: int = 220
    ALPHA_MEDIUM: int = 150
    ALPHA_LOW: int = 80
    ALPHA_SUBTLE: int = 40


COLORS = ColorPalette()


def get_color(name: str) -> Tuple[int, int, int]:

    color_map = {
        'cyan': COLORS.NEON_CYAN,
        'neon_cyan': COLORS.NEON_CYAN,
        'purple': COLORS.ELECTRIC_PURPLE,
        'electric_purple': COLORS.ELECTRIC_PURPLE,
        'pink': COLORS.HOT_PINK,
        'hot_pink': COLORS.HOT_PINK,
        'white': COLORS.TEXT_PRIMARY,
        'gray': COLORS.TEXT_SECONDARY,
        'green': COLORS.STATUS_OK,
        'orange': COLORS.STATUS_WARN,
        'red': COLORS.STATUS_DANGER,
        'bg': COLORS.DARK_BG,
        'panel': COLORS.PANEL_BG,
    }
    return color_map.get(name.lower(), COLORS.NEON_CYAN)


class GlowEffect:


    def __init__(self) -> None:
        config = get_config()
        glow_config = config.hud.glow

        self.enabled = glow_config.enabled
        self.layers = glow_config.layers


        self.layer_params: List[Dict] = []

        if self.layers >= 3:
            self.layer_params = [
                {'thickness': glow_config.outer_thickness, 'alpha': glow_config.outer_alpha},
                {'thickness': glow_config.middle_thickness, 'alpha': glow_config.middle_alpha},
                {'thickness': glow_config.inner_thickness, 'alpha': glow_config.inner_alpha},
            ]
        elif self.layers == 2:
            self.layer_params = [
                {'thickness': glow_config.outer_thickness, 'alpha': glow_config.middle_alpha},
                {'thickness': glow_config.inner_thickness, 'alpha': glow_config.inner_alpha},
            ]
        else:
            self.layer_params = [
                {'thickness': glow_config.inner_thickness, 'alpha': glow_config.inner_alpha},
            ]


        self._overlay_buffer: Optional[np.ndarray] = None
        self._buffer_shape: Tuple[int, int] = (0, 0)

    def _ensure_buffer(self, height: int, width: int) -> np.ndarray:

        if self._overlay_buffer is None or self._buffer_shape != (height, width):
            self._overlay_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self._buffer_shape = (height, width)
        else:
            self._overlay_buffer.fill(0)
        return self._overlay_buffer

    def draw_glowing_box(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        corner_radius: int = 0
    ) -> np.ndarray:

        if not self.enabled:
            cv2.rectangle(frame, pt1, pt2, color or COLORS.NEON_CYAN, 2)
            return frame

        if color is None:
            color = COLORS.NEON_CYAN

        height, width = frame.shape[:2]


        for layer in self.layer_params:
            thickness = layer['thickness']
            alpha = layer['alpha']

            if alpha >= 1.0:

                if corner_radius > 0:
                    self._draw_rounded_rect(frame, pt1, pt2, color, thickness, corner_radius)
                else:
                    cv2.rectangle(frame, pt1, pt2, color, thickness)
            else:

                overlay = self._ensure_buffer(height, width)

                if corner_radius > 0:
                    self._draw_rounded_rect(overlay, pt1, pt2, color, thickness, corner_radius)
                else:
                    cv2.rectangle(overlay, pt1, pt2, color, thickness)


                cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)

        return frame

    def _draw_rounded_rect(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        radius: int
    ) -> None:

        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

        if r <= 0:
            cv2.rectangle(frame, pt1, pt2, color, thickness)
            return


        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)


        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def draw_glowing_line(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:

        if not self.enabled:
            cv2.line(frame, pt1, pt2, color or COLORS.NEON_CYAN, 2)
            return frame

        if color is None:
            color = COLORS.NEON_CYAN

        height, width = frame.shape[:2]

        for layer in self.layer_params:
            thickness = layer['thickness']
            alpha = layer['alpha']

            if alpha >= 1.0:
                cv2.line(frame, pt1, pt2, color, thickness)
            else:
                overlay = self._ensure_buffer(height, width)
                cv2.line(overlay, pt1, pt2, color, thickness)
                cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)

        return frame

    def draw_glowing_circle(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int] = None,
        filled: bool = False
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN

        height, width = frame.shape[:2]

        if not self.enabled:
            thickness = -1 if filled else 2
            cv2.circle(frame, center, radius, color, thickness)
            return frame

        for layer in self.layer_params:
            thickness = -1 if filled else layer['thickness']
            alpha = layer['alpha']

            if alpha >= 1.0:
                cv2.circle(frame, center, radius, color, thickness)
            else:
                overlay = self._ensure_buffer(height, width)
                cv2.circle(overlay, center, radius, color, thickness)
                cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)

        return frame


class ScanlineEffect:


    def __init__(self, width: int = 640, height: int = 480) -> None:
        config = get_config()

        self.enabled = config.hud.scanlines_enabled
        self.opacity = config.hud.scanlines_opacity
        self.spacing = config.hud.scanlines_spacing


        self._pattern: Optional[np.ndarray] = None
        self._pattern_size: Tuple[int, int] = (0, 0)

        if self.enabled:
            self._generate_pattern(width, height)

    def _generate_pattern(self, width: int, height: int) -> None:

        if self._pattern_size == (width, height):
            return


        self._pattern = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(0, height, self.spacing):
            self._pattern[y:y+1, :] = [30, 30, 30]

        self._pattern_size = (width, height)

    def apply(self, frame: np.ndarray) -> np.ndarray:

        if not self.enabled or self._pattern is None:
            return frame

        height, width = frame.shape[:2]


        if self._pattern_size != (width, height):
            self._generate_pattern(width, height)


        cv2.addWeighted(self._pattern, self.opacity, frame, 1.0, 0, frame)

        return frame


class CornerBrackets:


    def __init__(self, bracket_length: int = 15, bracket_thickness: int = 2) -> None:
        self.length = bracket_length
        self.thickness = bracket_thickness

    def draw(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN

        x1, y1 = pt1
        x2, y2 = pt2
        L = self.length
        T = self.thickness


        cv2.line(frame, (x1, y1), (x1 + L, y1), color, T)
        cv2.line(frame, (x1, y1), (x1, y1 + L), color, T)


        cv2.line(frame, (x2, y1), (x2 - L, y1), color, T)
        cv2.line(frame, (x2, y1), (x2, y1 + L), color, T)


        cv2.line(frame, (x1, y2), (x1 + L, y2), color, T)
        cv2.line(frame, (x1, y2), (x1, y2 - L), color, T)


        cv2.line(frame, (x2, y2), (x2 - L, y2), color, T)
        cv2.line(frame, (x2, y2), (x2, y2 - L), color, T)

        return frame


class TargetingReticle:


    def __init__(self) -> None:
        self._frame_counter = 0
        self._rotation_speed = 3.0

    def draw(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        size: int = 30,
        color: Tuple[int, int, int] = None,
        animate: bool = True
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN

        cx, cy = center

        if animate:
            self._frame_counter += 1


        cv2.circle(frame, center, size, color, 1)


        gap = size // 3

        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, 1)

        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, 1)


        if animate:
            import math
            angle = (self._frame_counter * self._rotation_speed) % 360
            angle_rad = math.radians(angle)

            for i in range(4):
                a = angle_rad + (i * math.pi / 2)
                inner_r = size * 0.7
                outer_r = size * 0.9

                x1 = int(cx + inner_r * math.cos(a))
                y1 = int(cy + inner_r * math.sin(a))
                x2 = int(cx + outer_r * math.cos(a))
                y2 = int(cy + outer_r * math.sin(a))

                cv2.line(frame, (x1, y1), (x2, y2), color, 2)


        cv2.circle(frame, center, 2, color, -1)

        return frame


class DataPanel:


    def __init__(self) -> None:
        self._overlay_buffer: Optional[np.ndarray] = None
        self._buffer_shape: Tuple[int, int] = (0, 0)

    def draw(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        bg_color: Tuple[int, int, int] = None,
        border_color: Tuple[int, int, int] = None,
        alpha: float = 0.7
    ) -> np.ndarray:

        if bg_color is None:
            bg_color = COLORS.PANEL_BG
        if border_color is None:
            border_color = COLORS.NEON_CYAN

        height, width = frame.shape[:2]


        if self._overlay_buffer is None or self._buffer_shape != (height, width):
            self._overlay_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self._buffer_shape = (height, width)
        else:
            self._overlay_buffer.fill(0)


        cv2.rectangle(self._overlay_buffer, pt1, pt2, bg_color, -1)


        cv2.addWeighted(self._overlay_buffer, alpha, frame, 1.0, 0, frame)


        cv2.rectangle(frame, pt1, pt2, border_color, 1)

        return frame


class ProgressBar:


    def draw(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        width: int,
        height: int,
        progress: float,
        color: Tuple[int, int, int] = None,
        bg_color: Tuple[int, int, int] = None
    ) -> np.ndarray:

        if color is None:
            color = COLORS.NEON_CYAN
        if bg_color is None:
            bg_color = COLORS.PANEL_BG

        x, y = pt1
        progress = max(0.0, min(1.0, progress))
        fill_width = int(width * progress)


        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)


        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)


        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)

        return frame


_glow_effect: Optional[GlowEffect] = None
_scanline_effect: Optional[ScanlineEffect] = None
_corner_brackets: Optional[CornerBrackets] = None
_targeting_reticle: Optional[TargetingReticle] = None
_data_panel: Optional[DataPanel] = None
_progress_bar: Optional[ProgressBar] = None


def get_glow_effect() -> GlowEffect:

    global _glow_effect
    if _glow_effect is None:
        _glow_effect = GlowEffect()
    return _glow_effect


def get_scanline_effect() -> ScanlineEffect:

    global _scanline_effect
    if _scanline_effect is None:
        _scanline_effect = ScanlineEffect()
    return _scanline_effect


def get_corner_brackets() -> CornerBrackets:

    global _corner_brackets
    if _corner_brackets is None:
        _corner_brackets = CornerBrackets()
    return _corner_brackets


def get_targeting_reticle() -> TargetingReticle:

    global _targeting_reticle
    if _targeting_reticle is None:
        _targeting_reticle = TargetingReticle()
    return _targeting_reticle


def get_data_panel() -> DataPanel:

    global _data_panel
    if _data_panel is None:
        _data_panel = DataPanel()
    return _data_panel


def get_progress_bar() -> ProgressBar:

    global _progress_bar
    if _progress_bar is None:
        _progress_bar = ProgressBar()
    return _progress_bar
