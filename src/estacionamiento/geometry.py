"""Geometría de cruce de una línea virtual.

El criterio es el del práctico de seguimiento de la materia: el centro de la caja
cambia de lado respecto de la recta y, en ese frame, queda cerca del segmento.
Cada track se cuenta una sola vez. El sentido del cambio de signo separa un
ingreso de un egreso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INGRESO = "ingreso"
EGRESO = "egreso"
POSITIVE_TO_NEGATIVE = "positive_to_negative"
NEGATIVE_TO_POSITIVE = "negative_to_positive"


def side_of_line(point, start, end) -> float:
    """Signo del producto cruzado entre el segmento y el punto.

    Positivo y negativo son los dos semiplanos. Un valor cercano a cero
    indica que el punto está sobre la recta.
    """
    px, py = point
    ax, ay = start
    bx, by = end
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def point_on_segment(point, start, end, eps: float = 2.0) -> bool:
    """True si el punto está a lo sumo a ``eps`` píxeles del segmento."""
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-6:
        return float(np.linalg.norm(point - start)) <= eps

    t = float(np.dot(point - start, segment) / length_sq)
    t = min(1.0, max(0.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection)) <= eps


def box_center(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    """Centro de la caja, el punto que se usa para la trayectoria y el cruce."""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


@dataclass(frozen=True)
class CrossingEvent:
    """Un vehículo cruzó la línea. ``delta`` es -1 al ingresar y +1 al egresar."""

    track_id: int
    kind: str
    delta: int


class LineCrossingCounter:
    """Memoria de lado y de tracks ya contados para una línea fija."""

    def __init__(
        self,
        line_start: tuple[float, float],
        line_end: tuple[float, float],
        dist_thresh: float = 18.0,
        entry_direction: str = POSITIVE_TO_NEGATIVE,
    ) -> None:
        if entry_direction not in (POSITIVE_TO_NEGATIVE, NEGATIVE_TO_POSITIVE):
            raise ValueError(
                "entry_direction debe ser 'positive_to_negative' o 'negative_to_positive'."
            )
        self.line_start = line_start
        self.line_end = line_end
        self.dist_thresh = dist_thresh
        self.entry_direction = entry_direction
        self._previous_side: dict[int, float] = {}
        self._counted: set[int] = set()
        self.ingresos = 0
        self.egresos = 0

    def update(self, track_id: int, center: tuple[float, float]) -> CrossingEvent | None:
        """Registra la posición actual del track y devuelve el evento, si hubo cruce."""
        if track_id < 0 or track_id in self._counted:
            return None

        current = side_of_line(center, self.line_start, self.line_end)
        previous = self._previous_side.get(track_id)
        self._previous_side[track_id] = current
        if previous is None:
            return None

        crossed = (previous > 0 and current < 0) or (previous < 0 and current > 0)
        if not crossed:
            return None
        if not point_on_segment(center, self.line_start, self.line_end, self.dist_thresh):
            return None

        positive_to_negative = previous > 0 and current < 0
        is_entry = positive_to_negative == (self.entry_direction == POSITIVE_TO_NEGATIVE)
        kind = INGRESO if is_entry else EGRESO
        delta = -1 if is_entry else 1

        self._counted.add(track_id)
        if is_entry:
            self.ingresos += 1
        else:
            self.egresos += 1
        return CrossingEvent(track_id=track_id, kind=kind, delta=delta)
