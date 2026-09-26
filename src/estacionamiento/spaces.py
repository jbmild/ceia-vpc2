"""Cuenta plazas libres y ocupadas con el detector entrenado sobre PKLot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

EMPTY_COLOR = (0, 180, 0)
OCCUPIED_COLOR = (0, 0, 255)
WHITE = (255, 255, 255)


@dataclass
class SpaceSummary:
    frames: int
    empty: int
    occupied: int
    output: str


def load_spaces_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_space_labels(labels: list[str], empty_names: list[str], occupied_names: list[str]) -> dict[str, int]:
    empty_set = {name.lower() for name in empty_names}
    occupied_set = {name.lower() for name in occupied_names}
    empty = occupied = 0
    for label in labels:
        key = label.lower()
        if key in empty_set:
            empty += 1
        elif key in occupied_set:
            occupied += 1
    return {"empty": empty, "occupied": occupied}


def process_spaces(
    source: str | Path,
    output_path: str | Path,
    config: dict,
    frame_stride: int = 1,
    max_images: int = 0,
) -> SpaceSummary:
    """Anota una imagen, un video o una carpeta. El conteo del resumen es el del último frame."""
    source = Path(source)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(config["model"])
    names = model.names
    frames = list_frames(source, frame_stride, max_images)
    if not frames:
        raise FileNotFoundError(f"No hay imágenes en {source}")

    writer = None
    last_counts = {"empty": 0, "occupied": 0}
    saved = 0
    for frame in frames:
        annotated, last_counts = annotate_frame(model, names, frame, config)
        if writer is None:
            height, width = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                2.0 if source.is_dir() else 25.0,
                (width, height),
            )
        writer.write(annotated)
        saved += 1
    if writer is not None:
        writer.release()
    return SpaceSummary(
        frames=saved,
        empty=last_counts["empty"],
        occupied=last_counts["occupied"],
        output=str(output_path),
    )


def annotate_frame(model, names, frame, config):
    results = model.predict(
        frame,
        conf=float(config["conf"]),
        iou=float(config["iou"]),
        verbose=False,
    )
    labels = []
    annotated = frame.copy()
    boxes = results[0].boxes
    empty_names = {name.lower() for name in config["empty_names"]}
    if boxes is not None:
        for coords, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
            label = str(names.get(int(class_id), class_id))
            labels.append(label)
            color = EMPTY_COLOR if label.lower() in empty_names else OCCUPIED_COLOR
            x1, y1, x2, y2 = (int(value) for value in coords)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    counts = count_space_labels(labels, config["empty_names"], config["occupied_names"])
    text = f"Libres: {counts['empty']}   Ocupados: {counts['occupied']}"
    cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
    return annotated, counts


def list_frames(source: Path, frame_stride: int, max_images: int) -> list:
    if source.is_dir():
        images = sorted(
            path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if max_images and len(images) > max_images:
            step = max(1, len(images) // max_images)
            images = images[::step][:max_images]
        frames = []
        for path in images:
            image = cv2.imread(str(path))
            if image is not None:
                frames.append(image)
        return frames

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        image = cv2.imread(str(source))
        if image is None:
            return []
        return [image]
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_stride == 0:
            frames.append(frame)
            if max_images and len(frames) >= max_images:
                break
        index += 1
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuenta plazas libres y ocupadas.")
    parser.add_argument("--source", required=True, help="Imagen, video o carpeta de imágenes.")
    parser.add_argument("--output", default="outputs/plazas.mp4")
    parser.add_argument("--config", default="configs/spaces.yaml")
    parser.add_argument("--max-images", type=int, default=0, help="0 procesa todas. Útil para carpetas grandes.")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    summary = process_spaces(
        args.source,
        args.output,
        load_spaces_config(args.config),
        max_images=args.max_images,
    )
    print(json.dumps(asdict(summary), indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
