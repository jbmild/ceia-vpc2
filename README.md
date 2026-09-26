# Estacionamiento inteligente

Prototipo de visión por computadora para un estacionamiento. Con cámaras de vigilancia estima cuántas plazas están libres u ocupadas y, en el acceso, cuenta vehículos que ingresan o egresan.

Hay dos fuentes, las mismas del diagrama en `assets/`:

- Cámara interna: YOLO11 entrenado sobre PKLot marca cada plaza como libre u ocupada.
- Cámara de entrada y salida: YOLO11s preentrenado en COCO, ByteTrack y el cruce de una línea virtual. No se reentrena este detector.

## Por qué estos modelos

Las plazas se detectan, no se clasifica la imagen entera: un frame tiene decenas de lugares y hace falta una caja por lugar. YOLO11 es de una etapa, así que cierra detección y clase en una sola pasada. Un detector de dos etapas como Faster R-CNN suele localizar mejor en algunos benchmarks, pero suma una red de propuestas y deja menos margen para video en vivo. En esta RTX 4060 Ti la inferencia del detector de plazas queda cerca de 10 ms por frame.

Entre YOLO11n y YOLO11s se entrenó a los dos, 20 épocas, imágenes de 640 y batch 8, con el mismo corte de PKLot. El criterio de elección es el mayor mAP50 en el test entre los modelos cuya latencia no supera 1.5 veces la del más rápido. Ganó YOLO11n.

| Modelo | mAP50 test | mAP50-95 test | Precisión | Recall | Latencia |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.914 | 0.857 | 0.887 | 0.958 | 9.83 ms |
| YOLO11s | 0.891 | 0.856 | 0.889 | 0.958 | 9.52 ms |

En validación, durante el entrenamiento, los dos superan 0.99 de mAP50. El test son otros días de captura, y ahí el número baja. Ese corte evita inflar la métrica con fotos de cinco minutos después de las de entrenamiento.

En el acceso no hace falta un dataset propio: COCO ya incluye autos, camiones, colectivos y motos. ByteTrack mantiene un id por vehículo. Sin ese id, el mismo auto que cruza la línea en varios frames se contaría varias veces. El cambio de semiplano del centro de la caja, cerca del segmento, separa un ingreso (−1 al cupo) de un egreso (+1).

## Instalación

Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Si `torch.cuda.is_available()` da falso y el driver es CUDA 12.6, instalá el build que coincide con ese driver:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Dataset

PKLot (UFPR): imágenes de estacionamiento con cada plaza libre u ocupada. El export YOLO de Roboflow está en https://universe.roboflow.com/faress-workspace/pklot-6mfd3.

```bash
export ROBOFLOW_API_KEY=...
python scripts/download_dataset.py
```

Sin clave, el mismo script baja el archivo oficial (o su espejo en Hugging Face), convierte los polígonos a cajas y arma `data/pklot.yaml`. El corte es por día de captura, semilla 22: 8864 train, 1814 validación, 1738 test.

```bash
python scripts/download_dataset.py --source ufpr
```

El dataset, los pesos y los videos de salida no se versionan.

## Entrenamiento e inferencia

```bash
python scripts/train_spaces.py
python -m estacionamiento.spaces --source data/pklot/test/images --max-images 12
python -m estacionamiento.access --video ruta/al/acceso.mp4 --output outputs/acceso.mp4
python -m estacionamiento.demo \
  --access-video ruta/al/acceso.mp4 \
  --spaces-source data/pklot/test/images \
  --output outputs/demo.mp4
```

`scripts/train_spaces.py` escribe `results/spaces_metrics.json` y `configs/spaces.yaml`. La línea de conteo del acceso está en `configs/access.yaml`, en coordenadas del frame. Si cae fuera de la imagen, el script la reemplaza por una horizontal y lo avisa.

En la corrida de demostración, sobre el video de la clase de seguimiento, el acceso contó 3 ingresos y 1 egreso (19.17 ms por frame, seguimiento incluido). El último frame de la muestra de plazas marcó 25 libres y 15 ocupadas (24.29 ms por frame, dibujo incluido).

## Tests

```bash
pytest
```

Cubren el sentido del cruce, la conversión de una anotación PKLot a caja YOLO y el conteo de etiquetas libres y ocupadas.
