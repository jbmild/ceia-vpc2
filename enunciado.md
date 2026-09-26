# Pautas Trabajo Práctico – Visión por Computadora II – CEIA

## 1. Objetivo del Proyecto

El objetivo principal de este trabajo es el diseño y desarrollo de un prototipo funcional orientado a resolver una problemática del mundo real mediante el uso de modelos de visión por computadora basados en deep learning.

Se busca que los alumnos demuestren su capacidad para seleccionar, implementar y validar arquitecturas de redes neuronales, justificando sus decisiones técnicas basándose en métricas de rendimiento y experimentación.

## 2. Conformación de Equipos

- **Integrantes:** El trabajo es de carácter grupal
- **Tamaño:** Grupos de 4 personas

## 3. Requisitos Técnicos y Alcance

El proyecto debe tener como eje principal la aplicación de modelos pre-entrenados de visión por computadora.

### Eje Central (Obligatorio)

La solución debe implementar al menos una o más de las siguientes tareas principales:

- Clasificación de imágenes.
- Detección de objetos.
- Segmentación (Semántica o de Instancia).

### Capacidades Adicionales (Opcional)

Se valorará muy positivamente la integración de técnicas complementarias tales como:

- Estimación de pose (Pose Estimation).
- Reconocimiento óptico de caracteres (OCR).
- Estimación de profundidad (Depth Estimation).
- Seguimiento de objetos (Object Tracking).

## 4. Sistema de Evaluación

La calificación final se compone de tres pilares fundamentales, con sus respectivos pesos sobre la nota total:

| Entregable | Descripción | Peso |
| --- | --- | --- |
| Informe académico | Formato de conference paper IEEE | 30% |
| Repositorio GitHub | Código fuente, documentación y reproducibilidad. | 20% |
| Presentación y Demo | Exposición en vivo y video de demostración. | 50% |

## 5. Detalles de los Entregables

### A. Informe del Proyecto (Paper IEEE)

El informe debe redactarse siguiendo estrictamente el formato de conferencia de IEEE.

- **Extensión:** Entre 4 y 6 páginas (obligatorio).
- **Contenido esperado:**
  - Resumen (Abstract)
  - Introducción (Contexto del problema y del caso de estudio, objetivo del proyecto, al menos un caso real de la aplicación de un proyecto similar con antigüedad no mayor a 7 años)
  - Metodología, Diseño y Desarrollo
  - Resultados Experimentales y Análisis de resultados
  - Conclusiones (Al menos 5 conclusiones)
  - Bibliografía (Al menos 5 referencias)
- **Plantillas:** Disponibles en IEEE Manuscript Templates (Word A4 o LaTeX).
- **Enfoque:** Se deben priorizar las métricas y resultados obtenidos por sobre la teoría básica más que conocida, que ya hemos tocado y profundizado en clase. Enfocarse en lo nuevo que aporta su proyecto, no repetir tanto las cosas que ya hemos visto en clase.

### B. Repositorio de GitHub

El repositorio debe ser el centro de colaboración y entrega de código. Debe incluir:

- Código fuente limpio y comentado.
- Archivo README.md detallando el objetivo del proyecto e instrucciones de instalación.
- Archivo de requerimientos (`requirements.txt` o similar) para asegurar la reproducibilidad.
- Enlace al dataset utilizado o instrucciones para descargarlo.

### C. Presentación y Demostración Funcional

La defensa del proyecto se realizará de forma sincrónica con las siguientes condiciones:

- **Duración:** 12 minutos de exposición + 2 minutos de preguntas.
- **Participación:** Todos los integrantes del grupo deben exponer.
- **Demo en Video:** Para garantizar la fluidez de la presentación, se debe proyectar un video grabado previamente (máximo 2 minutos). Uno de los integrantes deberá narrar y explicar en vivo el funcionamiento del prototipo mientras se reproduce el video. Estos dos minutos están dentro de los 12 minutos de exposición.

## 6. Fecha de entrega

La exposición del trabajo se realizará en vivo durante la clase 8, el día **12 de octubre**. Y el plazo máximo para entregar el documento formato paper IEEE y el repositorio de github con el código es hasta las **23:59** de ese mismo día. En las próximas semanas se les enviará el link del formulario de envío.

## 7. Recomendaciones para el Éxito

1. **Justificación Técnica:** No se limiten a describir "qué" hizo, sino "por qué" lo hizo (ej. por qué eligió ResNet sobre EfficientNet, o YOLOv8 sobre Faster R-CNN).
2. **Métricas:** Respalde sus conclusiones con datos (mAP, F1-Score, latencia, etc.).
3. **Aplicación Real:** Asegúrese de que el problema atacado tenga una utilidad práctica clara y bien definida, y que ataque a una problemática real de la industria o nuestra sociedad.
