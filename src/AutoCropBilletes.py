"""
AutoCropBilletes.py
Funciones para recorte automático de billetes y división en cuadrantes.
"""

import cv2
import numpy as np


def _ordenar_puntos(pts):
    """Ordena 4 puntos en: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _transformar_perspectiva(image, pts):
    """Aplica transformación de perspectiva para enderezar el billete."""
    rect = _ordenar_puntos(pts)
    (tl, tr, br, bl) = rect

    anchura_abajo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    anchura_arriba = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_anchura = max(int(anchura_abajo), int(anchura_arriba))

    altura_derecha = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    altura_izquierda = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_altura = max(int(altura_derecha), int(altura_izquierda))

    destino = np.array([
        [0, 0],
        [max_anchura - 1, 0],
        [max_anchura - 1, max_altura - 1],
        [0, max_altura - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, destino)
    return cv2.warpPerspective(image, M, (max_anchura, max_altura))


def _generar_mapas_bordes(gris):
    """Genera múltiples mapas de bordes con distintas estrategias de preprocesamiento.
    Esto aumenta la probabilidad de detectar el contorno del billete incluso
    cuando está arrugado, doblado, o sobre fondos complejos."""
    mapas = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Estrategia 1: Canny con umbrales adaptativos (mediana) + Gaussian blur
    desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)
    v = np.median(desenfocada)
    bordes1 = cv2.Canny(desenfocada, int(max(0, 0.5 * v)), int(min(255, 1.5 * v)))
    bordes1 = cv2.dilate(bordes1, kernel, iterations=2)
    bordes1 = cv2.morphologyEx(bordes1, cv2.MORPH_CLOSE, kernel, iterations=2)
    mapas.append(bordes1)

    # Estrategia 2: Umbralización adaptativa (funciona mejor con fondos irregulares)
    thresh = cv2.adaptiveThreshold(desenfocada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    mapas.append(thresh)

    # Estrategia 3: Bilateral filter (preserva bordes) + Canny con umbrales más bajos
    bilateral = cv2.bilateralFilter(gris, 11, 75, 75)
    bordes3 = cv2.Canny(bilateral, 30, 100)
    bordes3 = cv2.dilate(bordes3, kernel, iterations=3)
    bordes3 = cv2.morphologyEx(bordes3, cv2.MORPH_CLOSE, kernel, iterations=3)
    mapas.append(bordes3)

    return mapas


def _buscar_contorno_rectangular(contornos, area_imagen, epsilons=[0.02, 0.03, 0.05]):
    """Busca el mejor contorno rectangular entre los contornos dados.
    Prueba múltiples valores de epsilon para la aproximación poligonal,
    lo que permite detectar rectángulos incluso con bordes irregulares."""
    for eps in epsilons:
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, eps * perimetro, True)
            if len(approx) == 4 and cv2.contourArea(approx) > area_imagen * 0.10:
                return approx
    return None


def auto_crop_billete(imagen):
    """
    Detecta automáticamente el contorno del billete en una imagen y lo recorta.

    ALGORITMO:
    1. Convertir a escala de grises
    2. Generar múltiples mapas de bordes con distintas técnicas:
       - Canny con umbrales adaptativos
       - Umbralización adaptativa gaussiana
       - Bilateral filter + Canny con umbrales bajos
    3. Para cada mapa, buscar contornos rectangulares (4 vértices)
       probando distintos niveles de aproximación poligonal
    4. Si se encuentra un rectángulo, aplicar transformación de perspectiva
    5. Fallback: usar minAreaRect (rectángulo rotado de área mínima)
       que funciona mejor con billetes arrugados/doblados

    Parámetros:
        imagen: numpy array (BGR) - La imagen de entrada con el billete.

    Retorna:
        numpy array (BGR) - La imagen recortada conteniendo solo el billete,
        o la imagen original si no se detectó un contorno rectangular.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    area_imagen = imagen.shape[0] * imagen.shape[1]

    # 1. Probar múltiples estrategias de detección de bordes
    mapas_bordes = _generar_mapas_bordes(gris)

    mejor_contorno = None
    mejor_area = 0

    for bordes in mapas_bordes:
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue

        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

        # Buscar contorno rectangular con distintos niveles de tolerancia
        rect = _buscar_contorno_rectangular(contornos, area_imagen)
        if rect is not None:
            area = cv2.contourArea(rect)
            if area > mejor_area:
                mejor_area = area
                mejor_contorno = rect

    # 2. Si encontramos un rectángulo, aplicar transformación de perspectiva
    if mejor_contorno is not None:
        puntos = mejor_contorno.reshape(4, 2)
        return _transformar_perspectiva(imagen, puntos)

    # 3. Fallback: usar minAreaRect (rectángulo rotado) del contorno más grande
    # Esto funciona mejor que boundingRect para billetes rotados o arrugados
    todos_contornos = []
    for bordes in mapas_bordes:
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        todos_contornos.extend(contornos)

    if todos_contornos:
        mayor = max(todos_contornos, key=cv2.contourArea)
        if cv2.contourArea(mayor) > area_imagen * 0.10:
            rect_rotado = cv2.minAreaRect(mayor)
            box = cv2.boxPoints(rect_rotado)
            box = box.astype("float32")
            return _transformar_perspectiva(imagen, box)

    return imagen


def dividir_en_cuadrantes(imagen, n_filas=2, n_cols=2):
    """
    Divide una imagen en una grilla de n_filas x n_cols cuadrantes.

    JUSTIFICACIÓN DE 4 CUADRANTES (2x2):
    Los billetes tienen una relación de aspecto ~2:1. Dividirlos en una grilla 2x2
    genera 4 regiones con aspectos razonables que capturan zonas distintas:
    - Superior-Izquierdo: generalmente contiene elementos decorativos o el retrato
    - Superior-Derecho: suele tener la denominación o marcas de agua
    - Inferior-Izquierdo: patrones de seguridad y textos
    - Inferior-Derecho: elementos decorativos y números

    Al comparar cuadrante-a-cuadrante, se reduce la ambigüedad porque cada región
    tiene características más distintivas que el billete completo.

    Parámetros:
        imagen: numpy array (BGR o gris) - La imagen a dividir.
        n_filas: int - Número de filas de la grilla (default: 2).
        n_cols: int - Número de columnas de la grilla (default: 2).

    Retorna:
        list[numpy array] - Lista de sub-imágenes (cuadrantes), ordenados de
        izquierda a derecha, de arriba hacia abajo (Q0=top-left, Q1=top-right,
        Q2=bottom-left, Q3=bottom-right).
    """
    alto, ancho = imagen.shape[:2]
    alto_q = alto // n_filas
    ancho_q = ancho // n_cols

    cuadrantes = []
    for fila in range(n_filas):
        for col in range(n_cols):
            y0 = fila * alto_q
            y1 = (fila + 1) * alto_q if fila < n_filas - 1 else alto
            x0 = col * ancho_q
            x1 = (col + 1) * ancho_q if col < n_cols - 1 else ancho
            cuadrantes.append(imagen[y0:y1, x0:x1])

    return cuadrantes


def cargar_y_recortar(ruta_imagen):
    """Carga una imagen y aplica el recorte automático. Retorna None si falla."""
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"  [ERROR] No se pudo cargar: {ruta_imagen}")
        return None
    return auto_crop_billete(imagen)
