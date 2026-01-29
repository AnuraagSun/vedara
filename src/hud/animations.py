import time
import math
from typing import Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.config_loader import get_config


class EasingType(Enum):

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


@dataclass
class AnimationState:

    start_value: float = 0.0
    end_value: float = 1.0
    current_value: float = 0.0
    start_time: float = 0.0
    duration: float = 0.15
    easing: EasingType = EasingType.EASE_OUT
    is_complete: bool = False
    is_active: bool = False


class EasingFunctions:


    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in(t: float) -> float:
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        return 1.0 - (1.0 - t) * (1.0 - t)

    @staticmethod
    def ease_in_out(t: float) -> float:
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0

    @staticmethod
    def bounce(t: float) -> float:
        if t < 1.0 / 2.75:
            return 7.5625 * t * t
        elif t < 2.0 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    @staticmethod
    def elastic(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))

    @staticmethod
    def get_function(easing: EasingType) -> Callable[[float], float]:

        functions = {
            EasingType.LINEAR: EasingFunctions.linear,
            EasingType.EASE_IN: EasingFunctions.ease_in,
            EasingType.EASE_OUT: EasingFunctions.ease_out,
            EasingType.EASE_IN_OUT: EasingFunctions.ease_in_out,
            EasingType.BOUNCE: EasingFunctions.bounce,
            EasingType.ELASTIC: EasingFunctions.elastic,
        }
        return functions.get(easing, EasingFunctions.linear)


class Animator:


    def __init__(self) -> None:
        self._animations: Dict[str, AnimationState] = {}
        config = get_config()
        self._default_duration = config.hud.slide_duration_ms / 1000.0
        self._fade_duration = config.hud.fade_duration_ms / 1000.0

    def start(
        self,
        name: str,
        start_value: float,
        end_value: float,
        duration: float = None,
        easing: EasingType = EasingType.EASE_OUT
    ) -> None:

        if duration is None:
            duration = self._default_duration

        self._animations[name] = AnimationState(
            start_value=start_value,
            end_value=end_value,
            current_value=start_value,
            start_time=time.perf_counter(),
            duration=duration,
            easing=easing,
            is_complete=False,
            is_active=True
        )

    def update(self, name: str) -> float:

        if name not in self._animations:
            return 0.0

        anim = self._animations[name]

        if not anim.is_active or anim.is_complete:
            return anim.current_value


        elapsed = time.perf_counter() - anim.start_time
        t = min(1.0, elapsed / anim.duration)


        easing_func = EasingFunctions.get_function(anim.easing)
        eased_t = easing_func(t)


        anim.current_value = anim.start_value + (anim.end_value - anim.start_value) * eased_t


        if t >= 1.0:
            anim.is_complete = True
            anim.current_value = anim.end_value

        return anim.current_value

    def get_value(self, name: str, default: float = 0.0) -> float:

        if name not in self._animations:
            return default
        return self._animations[name].current_value

    def is_complete(self, name: str) -> bool:

        if name not in self._animations:
            return True
        return self._animations[name].is_complete

    def is_active(self, name: str) -> bool:

        if name not in self._animations:
            return False
        return self._animations[name].is_active

    def stop(self, name: str) -> None:

        if name in self._animations:
            del self._animations[name]

    def clear_all(self) -> None:

        self._animations.clear()


class SlideAnimation:


    def __init__(self, animator: Animator = None) -> None:
        self._animator = animator or Animator()

    def slide_in_from_left(
        self,
        name: str,
        target_x: int,
        start_offset: int = 100,
        duration: float = 0.15
    ) -> None:

        self._animator.start(
            name,
            start_value=float(target_x - start_offset),
            end_value=float(target_x),
            duration=duration,
            easing=EasingType.EASE_OUT
        )

    def slide_in_from_right(
        self,
        name: str,
        target_x: int,
        start_offset: int = 100,
        duration: float = 0.15
    ) -> None:

        self._animator.start(
            name,
            start_value=float(target_x + start_offset),
            end_value=float(target_x),
            duration=duration,
            easing=EasingType.EASE_OUT
        )

    def slide_in_from_top(
        self,
        name: str,
        target_y: int,
        start_offset: int = 50,
        duration: float = 0.15
    ) -> None:

        self._animator.start(
            name,
            start_value=float(target_y - start_offset),
            end_value=float(target_y),
            duration=duration,
            easing=EasingType.EASE_OUT
        )

    def slide_in_from_bottom(
        self,
        name: str,
        target_y: int,
        start_offset: int = 50,
        duration: float = 0.15
    ) -> None:

        self._animator.start(
            name,
            start_value=float(target_y + start_offset),
            end_value=float(target_y),
            duration=duration,
            easing=EasingType.EASE_OUT
        )

    def get_position(self, name: str, default: float = 0.0) -> int:

        return int(self._animator.update(name))

    def is_complete(self, name: str) -> bool:

        return self._animator.is_complete(name)


class FadeAnimation:


    def __init__(self, animator: Animator = None) -> None:
        self._animator = animator or Animator()

    def fade_in(self, name: str, duration: float = 0.1) -> None:

        self._animator.start(
            name,
            start_value=0.0,
            end_value=1.0,
            duration=duration,
            easing=EasingType.EASE_OUT
        )

    def fade_out(self, name: str, duration: float = 0.1) -> None:

        self._animator.start(
            name,
            start_value=1.0,
            end_value=0.0,
            duration=duration,
            easing=EasingType.EASE_IN
        )

    def get_alpha(self, name: str, default: float = 1.0) -> float:

        if not self._animator.is_active(name):
            return default
        return self._animator.update(name)

    def is_complete(self, name: str) -> bool:

        return self._animator.is_complete(name)


class PulseAnimation:


    def __init__(self, frequency: float = 2.0) -> None:
        self._frequency = frequency
        self._start_time = time.perf_counter()

    def get_value(self, min_val: float = 0.5, max_val: float = 1.0) -> float:

        elapsed = time.perf_counter() - self._start_time
        t = (math.sin(elapsed * self._frequency * 2 * math.pi) + 1) / 2
        return min_val + (max_val - min_val) * t

    def get_alpha(self) -> float:

        return self.get_value(0.5, 1.0)

    def get_scale(self) -> float:

        return self.get_value(0.95, 1.05)


class DetectionCardAnimator:


    def __init__(self) -> None:
        self._animator = Animator()
        self._slide = SlideAnimation(self._animator)
        self._fade = FadeAnimation(self._animator)
        self._active_cards: Dict[int, float] = {}
        self._card_lifetime = 0.5

    def on_detection_appear(self, detection_id: int, target_x: int) -> None:

        slide_name = f"card_slide_{detection_id}"
        fade_name = f"card_fade_{detection_id}"

        if not self._animator.is_active(slide_name):
            self._slide.slide_in_from_right(slide_name, target_x, start_offset=80)
            self._fade.fade_in(fade_name, duration=0.1)
            self._active_cards[detection_id] = time.perf_counter()

    def on_detection_disappear(self, detection_id: int) -> None:

        fade_name = f"card_fade_{detection_id}"

        if self._animator.is_active(fade_name):
            self._fade.fade_out(fade_name, duration=0.15)

    def get_card_position(self, detection_id: int, default_x: int) -> int:

        slide_name = f"card_slide_{detection_id}"
        return self._slide.get_position(slide_name, default=float(default_x))

    def get_card_alpha(self, detection_id: int) -> float:

        fade_name = f"card_fade_{detection_id}"
        return self._fade.get_alpha(fade_name, default=1.0)

    def cleanup_old_cards(self, active_ids: set) -> None:

        to_remove = []

        for card_id in self._active_cards:
            if card_id not in active_ids:
                to_remove.append(card_id)

        for card_id in to_remove:
            self._animator.stop(f"card_slide_{card_id}")
            self._animator.stop(f"card_fade_{card_id}")
            del self._active_cards[card_id]


_animator: Optional[Animator] = None
_detection_card_animator: Optional[DetectionCardAnimator] = None
_pulse: Optional[PulseAnimation] = None


def get_animator() -> Animator:

    global _animator
    if _animator is None:
        _animator = Animator()
    return _animator


def get_detection_card_animator() -> DetectionCardAnimator:

    global _detection_card_animator
    if _detection_card_animator is None:
        _detection_card_animator = DetectionCardAnimator()
    return _detection_card_animator


def get_pulse() -> PulseAnimation:

    global _pulse
    if _pulse is None:
        _pulse = PulseAnimation()
    return _pulse
