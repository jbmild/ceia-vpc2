# Estacionamiento inteligente

Prototipo de visión por computadora para un estacionamiento. A partir de cámaras de vigilancia estima cuántas plazas están libres u ocupadas y, en la entrada y la salida, cuenta vehículos que ingresan o egresan.

El sistema usa dos fuentes de información:

- Cámaras internas: un detector entrenado sobre PKLot marca cada plaza como libre u ocupada.
- Cámara de acceso: un detector de vehículos preentrenado, seguimiento y el cruce de una línea virtual distinguen un ingreso de un egreso.

## Instalación

Requiere Python 3.10 o superior. Con una GPU NVIDIA el entrenamiento y la inferencia usan CUDA si PyTorch la detecta.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

La clave de Roboflow, si hace falta descargar PKLot, se lee de la variable de entorno `ROBOFLOW_API_KEY`. No la guardes en el repositorio.
