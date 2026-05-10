"""
demo_transformaciones.py
Genera un output visual de cada transformación aplicada por AutoCropBilletes.py
al recortar un billete. Guarda el resultado como una imagen compuesta.
"""

import cv2
import numpy as np
import sys
import os

# Añadir src al path para importar AutoCropBilletes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from AutoCropBilletes import _ordenar_puntos, _transformar_perspectiva, _generar_mapas_bordes, _buscar_contorno_rectangular


# ─── Configuración ────────────────────────────────────────────────────────────
RUTA_IMAGEN = os.path.join(os.path.dirname(__file__),
                           "data", "test_images", "my_messy_100_peso_photo.jpg")
RUTA_SALIDA = os.path.join(os.path.dirname(__file__), "demo_output.jpg")
# ──────────────────────────────────────────────────────────────────────────────


def texto(img, msg, pos=(10, 30), escala=0.7, color=(255, 255, 255)):
    """Añade texto con sombra para mejor legibilidad."""
    cv2.putText(img, msg, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_DUPLEX,
                escala, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_DUPLEX,
                escala, color, 1, cv2.LINE_AA)


def escalar(img, ancho_objetivo=520):
    """Escala la imagen manteniendo aspecto."""
    h, w = img.shape[:2]
    ratio = ancho_objetivo / w
    return cv2.resize(img, (ancho_objetivo, int(h * ratio)))


def a_bgr(img):
    """Convierte imagen en escala de grises o binaria a BGR para montaje."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def agregar_titulo(img, titulo, subtitulo=""):
    """Agrega una barra de título en la parte superior de la imagen."""
    h, w = img.shape[:2]
    barra = np.zeros((55, w, 3), dtype=np.uint8)
    barra[:] = (30, 30, 30)
    cv2.putText(barra, titulo, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                (220, 220, 220), 1, cv2.LINE_AA)
    if subtitulo:
        cv2.putText(barra, subtitulo, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (140, 180, 255), 1, cv2.LINE_AA)
    return np.vstack([barra, img])


def main():
    # ── 0. Cargar imagen ──────────────────────────────────────────────────────
    imagen = cv2.imread(RUTA_IMAGEN)
    if imagen is None:
        print(f"[ERROR] No se pudo cargar: {RUTA_IMAGEN}")
        sys.exit(1)

    print(f"[OK] Imagen cargada: {imagen.shape[1]}x{imagen.shape[0]} px")

    # ── 1. Escala de grises ───────────────────────────────────────────────────
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    area_imagen = imagen.shape[0] * imagen.shape[1]

    # ── 2. Generar mapas de bordes (las 3 estrategias) ────────────────────────
    mapas = _generar_mapas_bordes(gris)
    nombres_mapas = [
        ("Estrategia 1", "Canny adaptativo (mediana) + GaussianBlur + Dilation"),
        ("Estrategia 2", "Umbral adaptativo gaussiano + Morph Close"),
        ("Estrategia 3", "Bilateral Filter + Canny suave + Dilation"),
    ]

    # ── 3. Buscar contornos y el mejor rectángulo ─────────────────────────────
    mejor_contorno = None
    mejor_area = 0
    mejor_estrategia = -1

    panel_contornos = []

    for idx, bordes in enumerate(mapas):
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = a_bgr(bordes.copy())

        if contornos:
            contornos_sorted = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
            # Dibujar todos los candidatos en gris
            cv2.drawContours(vis, contornos_sorted, -1, (100, 100, 100), 1)

            rect = _buscar_contorno_rectangular(contornos_sorted, area_imagen)
            if rect is not None:
                area = cv2.contourArea(rect)
                # Dibujar el rectángulo detectado en verde
                cv2.drawContours(vis, [rect], -1, (0, 255, 80), 2)
                if area > mejor_area:
                    mejor_area = area
                    mejor_contorno = rect
                    mejor_estrategia = idx
                texto(vis, f"Rect OK  Area={int(area/1000)}K px2",
                      color=(80, 255, 80))
            else:
                texto(vis, "Sin rect 4-vertices", color=(80, 150, 255))
        else:
            texto(vis, "Sin contornos", color=(80, 80, 255))

        panel_contornos.append(vis)

    # ── 4. Dibujar contorno ganador sobre la imagen original ──────────────────
    img_contorno = imagen.copy()
    if mejor_contorno is not None:
        cv2.drawContours(img_contorno, [mejor_contorno], -1, (0, 255, 80), 3)
        texto(img_contorno, f"Contorno final (Estrat. {mejor_estrategia+1})",
              color=(80, 255, 80))
    else:
        texto(img_contorno, "Fallback: minAreaRect", color=(80, 150, 255))
        # Fallback: minAreaRect
        todos = []
        for b in mapas:
            c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            todos.extend(c)
        if todos:
            mayor = max(todos, key=cv2.contourArea)
            rect_rotado = cv2.minAreaRect(mayor)
            box = np.intp(cv2.boxPoints(rect_rotado))
            cv2.drawContours(img_contorno, [box], -1, (0, 150, 255), 3)
            mejor_contorno = box.astype("float32")

    # ── 5. Transformación de perspectiva (resultado final) ────────────────────
    if mejor_contorno is not None:
        pts = mejor_contorno.reshape(4, 2)
        recortado = _transformar_perspectiva(imagen, pts.astype("float32"))
    else:
        recortado = imagen.copy()

    # ── 6. Visualización de los 4 puntos ordenados ────────────────────────────
    img_puntos = imagen.copy()
    if mejor_contorno is not None:
        rect_ord = _ordenar_puntos(mejor_contorno.reshape(4, 2).astype("float32"))
        etiquetas = ["TL", "TR", "BR", "BL"]
        colores = [(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
        for i, (pt, lbl, col) in enumerate(zip(rect_ord, etiquetas, colores)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img_puntos, (x, y), 10, col, -1)
            texto(img_puntos, lbl, (x + 12, y + 8), escala=0.6, color=col)
        # Dibujar líneas de la perspectiva
        for i in range(4):
            p1 = tuple(rect_ord[i].astype(int))
            p2 = tuple(rect_ord[(i+1) % 4].astype(int))
            cv2.line(img_puntos, p1, p2, (255, 255, 0), 2)

    # ── 7. Construir el panel compuesto ───────────────────────────────────────
    W = 520  # ancho por panel

    def panel(img, titulo, sub=""):
        img_sc = escalar(a_bgr(img), W)
        return agregar_titulo(img_sc, titulo, sub)

    # Fila 1: original | gris
    orig_panel  = panel(imagen,       "0. Imagen original")
    gris_panel  = panel(gris,         "1. Escala de grises",
                        "cv2.cvtColor(BGR2GRAY)")

    # Fila 2: 3 mapas de bordes
    e1 = panel(mapas[0], nombres_mapas[0][0], nombres_mapas[0][1])
    e2 = panel(mapas[1], nombres_mapas[1][0], nombres_mapas[1][1])
    e3 = panel(mapas[2], nombres_mapas[2][0], nombres_mapas[2][1])

    # Fila 3: contornos detectados por estrategia
    c1 = panel(panel_contornos[0], "Contornos Estrat. 1", "approxPolyDP sobre bordes E1")
    c2 = panel(panel_contornos[1], "Contornos Estrat. 2", "approxPolyDP sobre bordes E2")
    c3 = panel(panel_contornos[2], "Contornos Estrat. 3", "approxPolyDP sobre bordes E3")

    # Fila 4: mejor contorno | puntos ordenados | resultado final
    cont_panel  = panel(img_contorno, "5. Mejor contorno detectado",
                        f"Area = {int(mejor_area/1000)}K px2")
    puntos_panel = panel(img_puntos,  "6. Puntos ordenados",
                        "TL, TR, BR, BL -> getPerspectiveTransform")
    final_panel  = panel(recortado,   "7. Resultado final",
                        "warpPerspective -> billete enderezado")

    def fila(*imgs):
        """Pega imágenes horizontalmente ajustando alturas."""
        max_h = max(i.shape[0] for i in imgs)
        padded = []
        for img in imgs:
            h, w = img.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        return np.hstack(padded)

    # Padding para fila de 2 (centrar)
    blank = np.zeros_like(orig_panel)

    fila1 = fila(orig_panel, gris_panel, blank)
    fila2 = fila(e1, e2, e3)
    fila3 = fila(c1, c2, c3)
    fila4 = fila(cont_panel, puntos_panel, final_panel)

    # Separador entre filas
    sep_color = (50, 50, 50)
    sep_h = 6

    def separador(ancho):
        s = np.zeros((sep_h, ancho, 3), dtype=np.uint8)
        s[:] = sep_color
        return s

    w_total = fila1.shape[1]
    compuesto = np.vstack([
        fila1, separador(w_total),
        fila2, separador(w_total),
        fila3, separador(w_total),
        fila4
    ])

    # Cabecera principal
    cabecera = np.zeros((70, w_total, 3), dtype=np.uint8)
    cabecera[:] = (20, 20, 40)
    cv2.putText(cabecera, "Pipeline AutoCropBilletes — Transformaciones paso a paso",
                (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(cabecera, os.path.basename(RUTA_IMAGEN),
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 180, 120), 1, cv2.LINE_AA)

    compuesto = np.vstack([cabecera, compuesto])

    cv2.imwrite(RUTA_SALIDA, compuesto, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK] Output guardado en: {RUTA_SALIDA}")
    print(f"     Tamaño final: {compuesto.shape[1]}x{compuesto.shape[0]} px")


if __name__ == "__main__":
    main()
