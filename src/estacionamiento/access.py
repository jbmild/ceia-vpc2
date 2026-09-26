"""Conteo de ingresos y egresos con un detector preentrenado y ByteTrack.

No se reentrena el modelo: YOLO11s, entrenado en COCO, propone vehículos y
ByteTrack les mantiene un id. El cruce de la línea virtual decide el sentido.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from estacionamiento.geometry import EGRESO, INGRESO, LineCrossingCounter, box_center

CYAN = (255, 255, 0)
RED = (0, 0, 255)
GREEN = (0, 180, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


@dataclass
class AccessSummary:
    ingresos: int
    egresos: int
    delta_cupo: int
    frames: int
    output: str


def _line_for_frame(line, width: int, height: int):
    """Usa la línea del config si cae dentro del frame. Si no, traza una horizontal usable."""
    start, end = line
    points = (*start, *end)
    inside = all(0 <= points[index] < width if index % 2 == 0 else 0 <= points[index] < height for index in range(4))
    if inside:
        return (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))
    y = int(height * 0.55)
    fitted = ((int(width * 0.15), y), (int(width * 0.85), y))
    print(
        f"La línea {start}->{end} queda fuera del frame {width}x{height}. "
        f"Se usa {fitted[0]}->{fitted[1]}."
    )
    return fitted


def load_access_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    start, end = config["line"]
    config["line"] = (tuple(start), tuple(end))
    return config


def process_access_video(
    video_path: str | Path,
    output_path: str | Path,
    config: dict,
) -> AccessSummary:
    """Procesa un video de acceso y escribe la versión anotada."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    line_start, line_end = _line_for_frame(config["line"], width, height)
    config = {**config, "line": (line_start, line_end)}
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(config["model"])
    names = model.names
    line_start, line_end = config["line"]
    counter = LineCrossingCounter(
        line_start,
        line_end,
        dist_thresh=float(config["dist_thresh"]),
        entry_direction=config["entry_direction"],
    )
    traces: dict[int, list[tuple[int, int]]] = {}
    kinds: dict[int, str] = {}
    target_classes = set(config["target_classes"])
    frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        cv2.line(frame, line_start, line_end, CYAN, 2)

        results = model.track(
            source=frame,
            conf=float(config["conf"]),
            iou=float(config["iou"]),
            tracker=config["tracker"],
            persist=True,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            for index, coords in enumerate(xyxy):
                class_name = names.get(int(classes[index]), "")
                if class_name not in target_classes:
                    continue
                x1, y1, x2, y2 = (int(value) for value in coords)
                track_id = int(track_ids[index]) if track_ids is not None else -1
                center = box_center(x1, y1, x2, y2)
                if track_id >= 0:
                    event = counter.update(track_id, center)
                    if event is not None:
                        kinds[track_id] = event.kind
                    _draw_trace(frame, traces, track_id, center, int(config["max_trace_len"]))
                _draw_box(frame, (x1, y1, x2, y2), center, class_name, track_id, kinds.get(track_id))

        _draw_hud(frame, counter)
        writer.write(frame)

    capture.release()
    writer.release()
    delta = counter.egresos - counter.ingresos
    return AccessSummary(
        ingresos=counter.ingresos,
        egresos=counter.egresos,
        delta_cupo=delta,
        frames=frames,
        output=str(output_path),
    )


def _draw_trace(frame, traces, track_id, center, max_len):
    history = traces.setdefault(track_id, [])
    history.append(center)
    if len(history) > max_len:
        del history[0]
    for previous, current in zip(history, history[1:]):
        cv2.line(frame, previous, current, YELLOW, 2)


def _draw_box(frame, box, center, class_name, track_id, kind):
    if kind == INGRESO:
        color = RED
    elif kind == EGRESO:
        color = GREEN
    else:
        color = CYAN
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, center, 4, color, -1)
    label = f"{class_name} id={track_id}"
    if kind:
        label = f"{label} {kind}"
    cv2.putText(frame, label, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_hud(frame, counter: LineCrossingCounter):
    text = f"Ingresos: {counter.ingresos}   Egresos: {counter.egresos}   Cupo: {counter.egresos - counter.ingresos:+d}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuenta ingresos y egresos en un video de acceso.")
    parser.add_argument("--video", required=True, help="Video de la cámara de entrada/salida.")
    parser.add_argument("--output", default="outputs/acceso.mp4", help="Video anotado de salida.")
    parser.add_argument("--config", default="configs/access.yaml")
    parser.add_argument("--summary", default="", help="JSON opcional con los conteos.")
    args = parser.parse_args()

    summary = process_access_video(args.video, args.output, load_access_config(args.config))
    print(json.dumps(asdict(summary), indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
