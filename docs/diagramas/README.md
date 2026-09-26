# Diagramas del sistema

Hay un diagrama de entrenamiento y uno de inferencia por cámara. Solo se entrena el detector de plazas. La cámara de acceso usa YOLO11s preentrenado en COCO sin reentrenar, así que solo tiene diagrama de inferencia.

| Archivo | Qué muestra | Código |
| --- | --- | --- |
| `01_entrenamiento_plazas.mmd` | Dataset PKLot desde Roboflow, fine-tuning de YOLO11n y YOLO11s, selección | `scripts/download_dataset.py`, `scripts/train_spaces.py` |
| `02_inferencia_plazas.mmd` | Cámara interna: plazas libres y ocupadas | `src/estacionamiento/spaces.py` |
| `03_inferencia_acceso.mmd` | Cámara de acceso: detección, seguimiento y cruce de línea | `src/estacionamiento/access.py`, `geometry.py` |

Tarea principal del enunciado: detección de objetos, en las dos cámaras. Técnica complementaria: seguimiento multiobjeto (ByteTrack) con conteo por cruce de línea.

Colores de los bloques: azul, datos de entrada; naranja, procesamiento; verde, modelos; violeta, archivos de salida; rojo, decisiones. El fondo de las etapas distingue las cámaras: verde en plazas, azul en acceso.

Para renderizar: pegar el contenido de un `.mmd` en https://mermaid.live, o usar `mmdc -i archivo.mmd -o archivo.png` con `@mermaid-js/mermaid-cli`.

## 1. Entrenamiento del detector de plazas

1. **Adquisición del dataset.** `scripts/download_dataset.py` descarga PKLot desde Roboflow (`faress-workspace/pklot-6mfd3`) por API, con `ROBOFLOW_API_KEY`. El export viene en formato YOLOv11: imágenes y un `.txt` por imagen con las cajas ya normalizadas, repartidas en train, valid y test según el corte de Roboflow.
2. **Preparación del dataset.** Se lee el `data.yaml` del export, se toman los nombres de clase (libre y ocupada) y se escriben las rutas absolutas de cada split en `data/pklot.yaml`, que es lo que consume el entrenamiento.
3. **Fine-tuning.** Se parte de los pesos COCO de YOLO11n y YOLO11s. Ultralytics redimensiona con letterbox a 640 px y aplica su aumentación por defecto (mosaico, HSV, volteo horizontal, escala y traslación). 20 épocas, batch 8, early stopping con paciencia 5. Se guarda `best.pt`, la mejor época sobre validación.
4. **Evaluación y selección.** Cada `best.pt` se evalúa sobre el split de test (mAP50, mAP50-95, precisión y recall). La latencia es el promedio de 40 imágenes tras 3 de calentamiento. Gana el mayor mAP50 entre los modelos cuya latencia no supera 1.5 veces la del más rápido.
5. **Artefactos.** `results/spaces_metrics.json` guarda las métricas de ambos. `configs/spaces.yaml` apunta al ganador con `conf 0.25` e `iou 0.5`; es el único vínculo entre el entrenamiento y la inferencia.

## 2. Inferencia de plazas (cámara interna)

1. **Entrada.** Una imagen, un video o una carpeta de imágenes, más `configs/spaces.yaml`.
2. **Preprocesamiento.** OpenCV lee los frames. En carpetas se submuestrea hasta `--max-images`; en video, cada `frame_stride` frames. Ultralytics aplica letterbox a 640 px, pasa de BGR a RGB, escala a [0, 1] y arma el tensor.
3. **Modelo principal.** Se cargan los pesos ya entrenados (`best.pt`, indicado en `configs/spaces.yaml`); en inferencia no se entrena nada. El detector marca cada plaza con su clase. Se descartan detecciones con confianza menor a 0.25 y NMS con IoU 0.5 elimina cajas duplicadas.
4. **Posprocesamiento.** Los nombres de clase se mapean a libre u ocupada (acepta variantes como `space-empty` u `ocupado`) y se cuentan por frame. Cada caja se dibuja en verde (libre) o rojo (ocupada), con el total arriba.
5. **Resultados.** Consola/terminal. Se genera `outputs/plazas.mp4` y se imprime un JSON con el conteo del último frame.

## 3. Inferencia de acceso (cámara de entrada y salida)

1. **Entrada.** Video de la entrada y `configs/access.yaml`, con la línea virtual y los umbrales.
2. **Preprocesamiento.** Antes de recorrer el video se verifica que la línea caiga dentro del frame; si no, se reemplaza por una horizontal al 55 % de la altura. Luego se lee frame a frame y Ultralytics aplica letterbox a 640 px.
3. **Modelo principal.** YOLO11s preentrenado en COCO, sin reentrenar. Confianza 0.4 y NMS con IoU 0.6.
4. **Modelo adicional: seguimiento.** ByteTrack asocia las detecciones entre frames y asigna un id a cada vehículo. Se conservan solo `car`, `truck`, `bus` y `motorcycle`.
5. **Posprocesamiento: cruce de línea.** Se mira de qué lado de la línea queda el centro de la caja (signo del producto cruzado). Hay cruce cuando el lado cambia respecto del frame anterior y el centro está a 18 px o menos del segmento. Cada id se cuenta una sola vez. Con la configuración por defecto, pasar de positivo a negativo es un ingreso (cupo −1) y al revés un egreso (cupo +1). La línea representa la barrera: hay que ubicarla sobre la barrera en el encuadre de la cámara.
6. **Anotación.** Se dibujan la línea, la trayectoria de los últimos 40 centros, la caja con clase e id (roja al ingresar, verde al egresar) y un HUD con ingresos, egresos y cupo.
7. **Resultados.** Consola/terminal. Se genera `outputs/acceso.mp4` y se imprime un JSON con ingresos, egresos y `delta_cupo`.

## Demo con las dos cámaras

`python -m estacionamiento.demo` ejecuta las dos inferencias una después de la otra, concatena los videos con una placa de título en `outputs/demo.mp4` e imprime un JSON con plazas libres, plazas ocupadas, ingresos, egresos, `delta_cupo` y milisegundos por frame (también en `results/demo_summary.json`). Los números solo se ponen lado a lado: los ingresos y egresos no corrigen el conteo de plazas ni al revés. No hay interfaz web ni de escritorio.
