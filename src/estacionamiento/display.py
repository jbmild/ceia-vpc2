"""Ventana opcional para ver las detecciones mientras se procesan."""

from __future__ import annotations

import time

import cv2

MAX_WINDOW_WIDTH = 1280


def show_frame(window: str, frame, delay_ms: int = 1) -> tuple[bool, float]:
    """Muestra el frame. Devuelve si hay que seguir y los segundos en pantalla.

    Esos segundos se descuentan de la latencia, que mide solo el procesamiento.
    """
    started = time.perf_counter()
    height, width = frame.shape[:2]
    if width > MAX_WINDOW_WIDTH:
        scale = MAX_WINDOW_WIDTH / width
        frame = cv2.resize(frame, (MAX_WINDOW_WIDTH, int(height * scale)))
    cv2.imshow(window, frame)
    keep_going = cv2.waitKey(max(1, delay_ms)) & 0xFF != ord("q")
    return keep_going, time.perf_counter() - started


def close_windows() -> None:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
