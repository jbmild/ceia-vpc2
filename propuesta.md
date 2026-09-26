# Estacionamiento inteligente

## Problema

El proyecto propone desarrollar un sistema basado en visión por computadora capaz de identificar en tiempo real los espacios de estacionamiento libres y ocupados a partir de imágenes obtenidas mediante cámaras de vigilancia.

Adicionalmente, se incorporará el monitoreo de los accesos al estacionamiento. Para esta funcionalidad no se entrenará un modelo específico, sino que se utilizará un modelo de detección de vehículos preentrenado, capaz de identificar automóviles en las imágenes de una cámara ubicada en la entrada y salida. Las detecciones se combinarán con técnicas de tracking y detección de cruce de una línea virtual para determinar si cada vehículo está ingresando o egresando del estacionamiento.

De esta manera, el sistema combinará dos fuentes de información:

- La cantidad de espacios libres y ocupados detectados mediante las cámaras internas.
- La cantidad de vehículos que ingresan o egresan detectados mediante un modelo preentrenado, tracking y line crossing.

## Dataset
https://universe.roboflow.com/faress-workspace/pklot-6mfd3

Para la detección de los espacios se utilizará un modelo entrenado sobre el dataset **PKLot**, que contiene imágenes de estacionamientos con plazas identificadas como libres (*empty*) u ocupadas (*occupied*).

## Diagrama lógico

![Diagrama lógico del sistema de estacionamiento inteligente](assets/diagrama%20logico.png)
