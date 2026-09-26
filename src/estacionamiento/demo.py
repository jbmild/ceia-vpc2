"""Junta la cámara de plazas y la de acceso en un video corto y un resumen."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from estacionamiento.access import load_access_config, process_access_video
from estacionamiento.spaces import load_spaces_config, process_spaces

WHITE = (255, 255, 255)


def run_demo(
    access_video: str | Path,
    spaces_source: str | Path,
    output_path: str | Path,
    access_config: str | Path = "configs/access.yaml",
    spaces_config: str | Path = "configs/spaces.yaml",
    max_images: int = 30,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    access_video_out = output_path.with_name("acceso.mp4")
    spaces_video_out = output_path.with_name("plazas.mp4")

    started = time.perf_counter()
    access_summary = process_access_video(access_video, access_video_out, load_access_config(access_config))
    access_ms = (time.perf_counter() - started) / max(access_summary.frames, 1) * 1000
    started = time.perf_counter()
    spaces_summary = process_spaces(
        spaces_source,
        spaces_video_out,
        load_spaces_config(spaces_config),
        max_images=max_images,
    )
    spaces_ms = (time.perf_counter() - started) / max(spaces_summary.frames, 1) * 1000
    write_figures(access_video_out, spaces_video_out)
    concatenate_videos(
        [
            ("Accesos: ingresos y egresos", access_video_out),
            ("Plazas libres y ocupadas", spaces_video_out),
        ],
        output_path,
    )
    summary = {
        "ingresos": access_summary.ingresos,
        "egresos": access_summary.egresos,
        "delta_cupo": access_summary.delta_cupo,
        "plazas_libres": spaces_summary.empty,
        "plazas_ocupadas": spaces_summary.occupied,
        "acceso_ms_por_frame": round(access_ms, 2),
        "plazas_ms_por_frame": round(spaces_ms, 2),
        "video": str(output_path),
    }
    results = Path("results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_figures(access_video: Path, spaces_video: Path) -> None:
    figures = Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)
    save_middle_frame(access_video, figures / "acceso.png")
    save_middle_frame(spaces_video, figures / "plazas.png")


def save_middle_frame(video_path: Path, image_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = capture.read()
    capture.release()
    if ok:
        cv2.imwrite(str(image_path), frame)


def concatenate_videos(parts: list[tuple[str, Path]], output_path: Path) -> None:
    clips = [(title, path) for title, path in parts if path.exists()]
    if not clips:
        raise FileNotFoundError("No hay videos para armar la demo.")
    probe = cv2.VideoCapture(str(clips[0][1]))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 20.0
    probe.release()

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for title, path in clips:
        for _ in range(int(fps)):
            writer.write(title_frame(title, width, height))
        capture = cv2.VideoCapture(str(path))
        clip_fps = capture.get(cv2.CAP_PROP_FPS) or fps
        repeat = max(1, round(fps / clip_fps))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (width, height))
            for _ in range(repeat):
                writer.write(resized)
        capture.release()
    writer.release()


def title_frame(title: str, width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, title, (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma el video de demostración del prototipo.")
    parser.add_argument("--access-video", required=True)
    parser.add_argument("--spaces-source", required=True)
    parser.add_argument("--output", default="outputs/demo.mp4")
    parser.add_argument("--access-config", default="configs/access.yaml")
    parser.add_argument("--spaces-config", default="configs/spaces.yaml")
    parser.add_argument("--max-images", type=int, default=30)
    args = parser.parse_args()
    summary = run_demo(
        args.access_video,
        args.spaces_source,
        args.output,
        args.access_config,
        args.spaces_config,
        args.max_images,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
