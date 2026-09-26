"""Descarga PKLot y deja un data yaml de YOLO en data/pklot.yaml.

Por defecto usa el export de Roboflow citado en la propuesta. Hace falta
ROBOFLOW_API_KEY. Si no hay clave, --source ufpr baja el archivo oficial de
UFPR (o su espejo) y convierte los polígonos a cajas YOLO. El corte es por
día de captura para no mezclar fotos casi iguales en train y test.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image

ROBOFLOW_WORKSPACE = "faress-workspace"
ROBOFLOW_PROJECT = "pklot-6mfd3"
UFPR_URLS = (
    "https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz",
    "https://web.inf.ufpr.br/vri/databases/PKLot.tar.gz",
    "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz",
)
CLASS_NAMES = ["empty", "occupied"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga PKLot en formato YOLO.")
    parser.add_argument("--source", choices=("roboflow", "ufpr"), default="roboflow")
    parser.add_argument("--version", type=int, default=1, help="Versión del dataset en Roboflow.")
    parser.add_argument("--destination", default="data")
    args = parser.parse_args()

    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    if args.source == "roboflow":
        dataset_dir = download_roboflow(destination, args.version)
        yaml_path = normalize_roboflow_yaml(dataset_dir, destination / "pklot.yaml")
    else:
        yaml_path = download_ufpr(destination)
    print(f"Dataset listo: {yaml_path}")


def download_roboflow(destination: Path, version: int) -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Falta ROBOFLOW_API_KEY. Exportá la clave o ejecutá de nuevo con --source ufpr."
        )
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(version).download("yolov11", location=str(destination / "roboflow"))
    return Path(dataset.location)


def normalize_roboflow_yaml(dataset_dir: Path, output_yaml: Path) -> Path:
    source_yaml = dataset_dir / "data.yaml"
    content = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = content.get("names", CLASS_NAMES)
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    splits = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = dataset_dir / folder / "images"
        if not images.exists():
            images = dataset_dir / ("val" if folder == "valid" else folder) / "images"
        if images.exists():
            splits[split] = str(images.resolve())
    if "val" not in splits:
        raise FileNotFoundError(f"No apareció el split de validación en {dataset_dir}")
    payload = {"path": str(dataset_dir.resolve()), "names": names, **splits}
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_yaml


def download_ufpr(destination: Path) -> Path:
    archive = destination / "PKLot.tar.gz"
    extracted = destination / "PKLot"
    if not any(extracted.rglob("*.xml")):
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            download_file(UFPR_URLS, archive)
        print(f"Extrayendo {archive}...")
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(destination, filter="data")
    return convert_ufpr_to_yolo(extracted if extracted.exists() else destination, destination)


def download_file(urls: tuple[str, ...], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        print(f"Descargando {url}")
        result = subprocess.run(
            ["curl", "-L", "--fail", "--retry", "2", "-C", "-", "-o", str(destination), url],
            check=False,
        )
        if result.returncode == 0 and destination.exists() and destination.stat().st_size > 1_000_000:
            return
        errors.append(f"{url} (código {result.returncode})")
    raise RuntimeError("No se pudo descargar PKLot: " + "; ".join(errors))


def convert_ufpr_to_yolo(source: Path, destination: Path) -> Path:
    samples = []
    for xml_path in sorted(source.rglob("*.xml")):
        image_path = xml_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        day = "/".join(xml_path.relative_to(source).parts[:-1])
        samples.append((image_path, xml_path, day))
    if not samples:
        raise FileNotFoundError(f"No hay anotaciones XML en {source}")

    days = sorted({day for _, _, day in samples})
    random.Random(22).shuffle(days)
    train_cut = int(len(days) * 0.70)
    val_cut = int(len(days) * 0.85)
    split_of_day = {}
    for index, day in enumerate(days):
        if index < train_cut:
            split_of_day[day] = "train"
        elif index < val_cut:
            split_of_day[day] = "valid"
        else:
            split_of_day[day] = "test"

    root = destination / "pklot"
    counts = {"train": 0, "valid": 0, "test": 0}
    for image_path, xml_path, day in samples:
        split = split_of_day[day]
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / image_path.name
        if not target.exists():
            target.symlink_to(image_path.resolve())
        boxes = parking_boxes(xml_path, image_path)
        lines = [f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for class_id, cx, cy, w, h in boxes]
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1

    payload = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }
    yaml_path = destination / "pklot.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("Imágenes por split:", counts)
    return yaml_path


def rotated_rect_points(space: ET.Element) -> list[tuple[float, float]]:
    center = space.find("./rotatedRect/center")
    size = space.find("./rotatedRect/size")
    angle = space.find("./rotatedRect/angle")
    if center is None or size is None or angle is None:
        return []
    cx = float(center.attrib["x"])
    cy = float(center.attrib["y"])
    width = float(size.attrib["w"])
    height = float(size.attrib["h"])
    theta = math.radians(float(angle.attrib.get("d", "0")))
    cosine, sine = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
        corners.append((cx + dx * cosine - dy * sine, cy + dx * sine + dy * cosine))
    return corners


def parking_boxes(xml_path: Path, image_path: Path) -> list[tuple[int, float, float, float, float]]:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes = []
    root = ET.parse(xml_path).getroot()
    for space in root.iter("space"):
        occupied = space.attrib.get("occupied", "0")
        class_id = 1 if occupied in {"1", "true", "True"} else 0
        points = [(float(point.attrib["x"]), float(point.attrib["y"])) for point in space.findall("./contour/point")]
        if len(points) < 2:
            points = rotated_rect_points(space)
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w < 2 or box_h < 2:
            continue
        cx = min(1.0, max(0.0, ((x1 + x2) / 2) / width))
        cy = min(1.0, max(0.0, ((y1 + y2) / 2) / height))
        boxes.append((class_id, cx, cy, min(1.0, box_w / width), min(1.0, box_h / height)))
    return boxes


if __name__ == "__main__":
    main()
