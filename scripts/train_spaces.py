"""Entrena YOLO11n y YOLO11s sobre PKLot y elige uno por mAP50 y latencia.

Se queda con el modelo de mayor mAP50 si su latencia no pasa de 1.5 veces la
del más rápido. En caso contrario se queda con el más rápido. Las métricas
van a results/spaces_metrics.json y el camino del ganador a configs/spaces.yaml.
Los pesos no se versionan.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

MODELS = ("yolo11n.pt", "yolo11s.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara YOLO11n y YOLO11s en PKLot.")
    parser.add_argument("--data", default="data/pklot.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-images", type=int, default=40)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"No está el dataset: {data_path}. Corré scripts/download_dataset.py primero.")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    project = Path("runs/spaces").resolve()
    reports = []
    for weights in MODELS:
        reports.append(train_and_measure(weights, args, project))

    fastest = min(report["latency_ms"] for report in reports)
    eligible = [report for report in reports if report["latency_ms"] <= fastest * 1.5]
    winner = max(eligible, key=lambda report: report["map50"])
    for report in reports:
        report["selected"] = report["model"] == winner["model"]

    payload = {
        "rule": "Mayor mAP50 entre los modelos cuya latencia no supera 1.5 veces la del más rápido.",
        "models": reports,
        "selected_model": winner["model"],
        "selected_weights": winner["weights"],
    }
    (results_dir / "spaces_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_spaces_config(winner["weights"])
    print(json.dumps(payload, indent=2))


def train_and_measure(weights: str, args, project: Path) -> dict:
    name = Path(weights).stem
    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        workers=4,
        device=args.device,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    best = project / name / "weights" / "best.pt"
    trained = YOLO(best)
    metrics = trained.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, plots=False)
    latency_ms = measure_latency(trained, args)
    return {
        "model": name,
        "weights": str(best),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "latency_ms": round(latency_ms, 2),
    }


def measure_latency(model: YOLO, args) -> float:
    content = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    images_dir = Path(content.get("test") or content["val"])
    if not images_dir.is_absolute():
        images_dir = Path(content.get("path", ".")) / images_dir
    images = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.latency_images]
    if not images:
        raise FileNotFoundError(f"No hay imágenes para medir latencia en {images_dir}")

    device = args.device
    use_cuda = str(device) != "cpu" and torch.cuda.is_available()
    for image in images[:3]:
        model.predict(str(image), imgsz=args.imgsz, device=device, verbose=False)
        if use_cuda:
            torch.cuda.synchronize()

    started = time.perf_counter()
    for image in images:
        model.predict(str(image), imgsz=args.imgsz, device=device, verbose=False)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (elapsed / len(images)) * 1000


def write_spaces_config(weights: str) -> None:
    weights_path = Path(weights)
    try:
        weights_path = weights_path.relative_to(Path.cwd())
    except ValueError:
        pass
    config = {
        "model": str(weights_path),
        "conf": 0.25,
        "iou": 0.5,
        "empty_names": ["empty", "space-empty", "libre", "vacant"],
        "occupied_names": ["occupied", "space-occupied", "ocupado"],
    }
    path = Path("configs/spaces.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
