# Manual de Usuario: Reconocimiento de Dinero Uruguayo

Este manual describe los pasos necesarios para configurar, entrenar y ejecutar el motor de reconocimiento de billetes uruguayos utilizando los algoritmos SIFT y ORB.

## Requisitos del Sistema

Para ejecutar el script `./src/MatchingFullBills.ipynb`, es necesario contar con un entorno de **Python 3.10+** con las siguientes bibliotecas instaladas:

* `opencv-python` (OpenCV): Para el procesamiento de imágenes y algoritmos de visión.
* `numpy`: Para el manejo de vectores de descriptores y cálculos numéricos.
* `glob`, `json`, `re`: Para la gestión de archivos y persistencia de datos.

## Estructura de Directorios

El sistema espera una estructura de carpetas específica para localizar las imágenes y almacenar la base de datos de características:

* `./data/bill_processing/processed_images/`: Contiene las imágenes de referencia (imágenes "limpias" y alineadas de cada cara de los billetes).
* `./data/test_images/`: Contiene las carpetas con las imágenes de prueba, organizadas por denominación (ej. `/100/`, `/500/`).
* `./data/full_bill_database/`: Directorio donde el sistema creará automáticamente las subcarpetas `sift_database` y `orb_database` para guardar los descriptores (`.npy`) y los umbrales entrenados (`thresholds.json`).

## Instrucciones de Ejecución

### 1. Preparación de Datos
Asegúrese de que las imágenes de referencia en `processed_images` sigan la nomenclatura esperada por el sistema (ej. `clean_100PesosFront.png`) para que la función de extracción de denominación funcione correctamente.

### 2. Entrenamiento y Primera Ejecución
Al ejecutar el bloque de **"Ejecución Principal"**, el sistema realizará las siguientes tareas de forma automática:

* Carga del Dataset.
* Generación de la Base de Datos.
* Cálculo de Umbrales.
* Predicciones y Resultados.

### 3. Evaluación de Resultados
El sistema mostrará:
* La denominación real vs. la predicción para cada imagen.
* El estado de la predicción (`CORRECTO` o `INCORRECTO`).
* La **Precisión Global (Accuracy)** final para cada algoritmo (SIFT y ORB).

## Ajuste de Parámetros

Si desea modificar el comportamiento del sistema, puede ajustar las constantes al inicio del script:

* `RATIO_THRESH`: Por defecto en `0.7`. Redúzcalo para ser más estricto con los *matches* o auméntelo para permitir más correspondencias.
* `nfeatures`: Definido en `2500` para ambos algoritmos. Este valor controla cuántos puntos clave se extraen de cada imagen.

