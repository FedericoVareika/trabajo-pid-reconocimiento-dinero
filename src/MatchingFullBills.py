import cv2
import os
import json
import glob
import numpy as np
import re
from collections import defaultdict
import random

import typing

# ------------------------------
# TIPOS PARA ASEGURAR INTEGRIDAD

# { category: [ filepath1.png, filepath2.png, ... ] }
DatasetFilepaths = dict[str, list[str]]

Algorithm = cv2.ORB | cv2.SIFT

Descriptors = cv2.typing.MatLike
KeyPoints = typing.Sequence[cv2.KeyPoint]
Feature = tuple[None | KeyPoints, None | Descriptors]

DistanceMatches = typing.Sequence[cv2.DMatch]

# ------------------------------

# -----------------------------------
# Configuración de rutas y parámetros

test_images_dir = "../data/test_images/"
png_dir = "../data/bill_processing/processed_images/"

database_dir = "../data/full_bill_database/"

orb_database_dir = "orb_database"
sift_database_dir = "sift_database"

RATIO_THRESH = 0.7
THRESHOLDS_FILES: dict[type, str] = {
        cv2.SIFT: f'{database_dir}{sift_database_dir}/thresholds.json',
        cv2.ORB: f'{database_dir}{orb_database_dir}/thresholds.json',
        }
# -----------------------------------

# -----------------------------------
# Helpers
def get_category_from_db_filename(filename: str) -> str:
    return filename.replace("clean_", "").replace(".png", "")

def get_denomination_from_category(category: str) -> str:
    match = re.search(r'\d+', category)
    if match:
        denom = match.group()
        if "Polimero" in category:
            return denom + "Polimero"
        return denom
    return category

def load_dataset(test_images_dir: str) -> DatasetFilepaths:
    dataset: DatasetFilepaths = defaultdict(list)
    for root, _, files in os.walk(test_images_dir):
        category = os.path.basename(root)
        if not category.isdigit():
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                dataset[category].append(filepath)
    return dataset

def split_dataset(dataset: DatasetFilepaths, train_ratio: float = 0.75):
    train_set: DatasetFilepaths = defaultdict(list)
    test_set: DatasetFilepaths = defaultdict(list)

    random.seed(42)
    for category, images in dataset.items():
        sorted_images = sorted(images)

        random.shuffle(sorted_images)

        split_idx = int(len(sorted_images) * train_ratio)

        train_set[category] = sorted_images[:split_idx]
        test_set[category] = sorted_images[split_idx:]

    return train_set, test_set
# -----------------------------------

def extract_features(
        image_path: str,
        algorithm: Algorithm) -> Feature:
    img = cv2.imread(image_path)
    if img is None:
        return None, None
        
    # Ajustar la dimension de las imagenes para tengan como maximo una dimension 
    # de 1024 pixeles (ancho o alto)
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Se utiliza la imagen en escala de grises para la detección de 
    # características porque algoritmos detectores de esquinas y bordes buscan
    # variaciones bruscas de intensidad luminosa. 
    # Las imágenes con color (RGB) triplican el costo computacional sin aportar
    # mejoras significativas a la estructura geométrica que define al billete.    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detección y descripción
    # --- ORB ---
    # kp = keypoints (puntos de interés detectados). Es una lista de puntos
    #     concretos (x,y) de las elecciones de ORB para trabajar. 
    #     Esta es la parte a la cual en la definición de ORB se le llama 
    #     FAST: para detectar donde están los puntos de interés (las esquinas).
    # des = descriptores (vectores numéricos de bits que describen las 
    #     características locales alrededor de cada keypoint). Esta es la parte
    #     a la cual en la definición de ORB se le llama BRIEF: para describir
    #     como son esos puntos (vector binario que representa el alrededor).
    # 
    # --- SIFT ---
    # kp = keypoints (puntos de interés detectados). En SIFT, esto se logra
    #     buscando extremos locales en una serie de imágenes desenfocadas 
    #     progresivamente (Diferencia de Gaussianas o DoG), lo que garantiza
    #     que los puntos sean invariantes a la escala.
    # des = descriptores. A diferencia del vector binario de ORB, SIFT genera 
    #     para cada punto un vector de 128 dimensiones de números de punto
    #     flotante. Este vector representa un histograma de las orientaciones 
    #     de los gradientes locales, haciéndolo invariante a la rotación.
    #
    kp, des = algorithm.detectAndCompute(gray, None)
    return kp, des

def match_is_good(closest_matches: DistanceMatches): 
    # Conteo de coincidencias mediante Ratio Test de Lowe
    # Aplicamos el "Ratio Test" de Lowe para filtrar falsos positivos ya que
    # ORB y SIFT pueden generar matches ambiguos. Al tener 2 puntos tan
    # distintos, signfica que el mejor match es significativamente mejor que el
    # segundo, lo que sugiere una correspondencia más confiable.
    # Parámetro RATIO_THRESH = 0.75: Valor estándar recomendado.
    # Lógica: Si la distancia al mejor match (m_n[0]) es menor al 75% de la
    # distancia al segundo mejor (m_n[1]), se considera una correspondencia 
    # robusta y no ambigua.
    return closest_matches[0].distance < RATIO_THRESH * closest_matches[1].distance

def count_good_matches(
        des_query: Descriptors,
        des_db: Descriptors,
        matcher: cv2.BFMatcher):

    # JUSTIFICACIÓN DEL MÉTODO KNN Y EL PARÁMETRO k=2
    # Se utiliza knnMatch() en lugar de match() simple o radiusMatch() porque
    # KNN es más adaptativo. No depende de un umbral de distancia fijo, como
    # radiusMatch() (que fallaría con cambios de iluminación), sino que
    # garantiza encontrar los vecinos más cercanos sin importar a qué distancia
    # absoluta se encuentren en el espacio del descriptor. 
    # Además, utilizamos k=2 porque el Ratio Test de Lowe requiere comparar el
    # mejor match con el segundo. No utilizamos 1 por lo ya explicado, ni mas 
    # de 2 porque agrega complejidad y no aporta beneficios para Lowe.
    matches = matcher.knnMatch(des_query, des_db, k=2)

    good_matches = filter(match_is_good, matches)
    return len(list(good_matches))

def find_intersection(pos_matches: list[int], neg_matches: list[int]) -> float:
    if not pos_matches or not neg_matches:
        return 10.0
    
    max_pos = float(max(pos_matches))
    max_neg = float(max(neg_matches))
    min_pos = float(min(pos_matches))
    
    if max_neg < min_pos:
        return (max_neg + min_pos) / 2.0
        
    max_val = max(max_pos, max_neg)
    bins = np.arange(0, max_val + 2)
    
    hist_pos, _ = np.histogram(pos_matches, bins=bins, density=True)
    hist_neg, _ = np.histogram(neg_matches, bins=bins, density=True)
    
    window = np.ones(3) / 3.0
    hist_pos = np.convolve(hist_pos, window, mode='same')
    hist_neg = np.convolve(hist_neg, window, mode='same')
    
    diff = hist_pos - hist_neg

    # Obtener los indices donde hay cambio de signo en las diferencias de 
    # positivos vs negativos.
    idx = np.argwhere(np.diff(np.sign(diff))).flatten()
    
    if len(idx) > 0:
        # Obtener el indice obtenido mas cercano al punto medio de los promedios 
        # de matches positivos y negativos.
        mid_mean = (np.mean(pos_matches) + np.mean(neg_matches)) / 2.0
        best_idx = min(idx, key=lambda i: abs(i - mid_mean))
        return float(best_idx)
    elif min_pos > max_neg:
        # Obtener un umbral por defecto si no hay cambios de signo
        return (max_neg + min_pos) / 2.0 
    else:
        return max_neg + 1.0

def get_db_features(
        png_dir: str,
        db_dir: str,
        algorithm: Algorithm) -> dict[str, Descriptors]:
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    db_features = {}
    
    npy_files = glob.glob(os.path.join(db_dir, "*.npy"))
    if npy_files:
        print(f"--- Cargando características precalculadas desde {db_dir} ---")
        for f in npy_files:
            des = np.load(f)
            png_name = os.path.basename(f).replace(".npy", ".png")
            db_features[png_name] = des
        return db_features
        
    print(f"--- Extrayendo y guardando características en {db_dir} ---")
    png_files = glob.glob(os.path.join(png_dir, "*.png"))
    for f in png_files:
        _, des = extract_features(f, algorithm)
        if des is not None:
            filename = os.path.basename(f)
            db_features[filename] = des
            npy_path = os.path.join(db_dir, filename.replace(".png", ".npy"))
            np.save(npy_path, des)
            
    return db_features

def train_thresholds(train_set: DatasetFilepaths, 
                     db_descriptors: dict[str, Descriptors],
                     algorithm: Algorithm) -> dict[str, float]:
    threshold_file = THRESHOLDS_FILES[type(algorithm)]
    if os.path.exists(threshold_file):
        print(f"--- Cargando umbrales desde {threshold_file} ---")
        with open(threshold_file, "r") as f:
            return json.load(f)
            
    print("--- Entrenando el modelo con imágenes completas (Histogramas) ---")
    
    train_features: dict[str, list[Feature]] = defaultdict(list)
    for cat, paths in train_set.items():
        for path in paths: 
            train_features[cat].append(extract_features(path, algorithm))

    #
    # Inicializamos el comparador por Fuerza Bruta. La justificación es que al
    # ser un dataset pequeño, el costo computacional es manejable. 
    # Se podría haber utilizado FLANN que es más rápido para grandes datasets
    # pero es menos preciso.
    # 
    # --- ORB ---
    # Parámetro normType=cv2.NORM_HAMMING: Obligatorio para descriptores 
    #   binarios como ORB ya que ORB devuelve un vector de binarios y este
    #   calcula la distancia basada en la diferencia de bits (XOR) en lugar de
    #   distancia euclidiana como SIFT.
    #
    # --- SIFT ---
    # Parámetro normType=cv2.NORM_L2: ES CRÍTICO cambiar esto respecto a ORB. 
    #   SIFT no usa vectores binarios, sino vectores de 128 dimensiones reales.
    #   NORM_L2 calcula la distancia Euclidiana tradicional entre estos 
    #   vectores, mientras que NORM_HAMMING fallaría por completo.
    # 
    # ------------ 
    # 
    # Parámetro crossCheck=False: Se desactiva para poder obtener los 2 vecinos
    #   más cercanos (k=2) en el paso KNN. Se explicará más adelante por qué se
    #   utiliza KNN.

    thresholds: dict[str, float] = {}
    matcher = None
    if (type(algorithm) is cv2.SIFT): 
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    for db_file, des_db in sorted(db_descriptors.items()):
        filename = os.path.basename(db_file)
        comp_category = get_category_from_db_filename(filename)
        denomination = get_denomination_from_category(comp_category)

        positive_matches: list[int] = []
        negative_matches: list[int] = []

        for cat, features in train_features.items():
            for _, des_query in features:

                if des_query is None or len(des_query) < 2 or len(des_db) < 2:
                    continue

                n_matches = count_good_matches(des_query, des_db, matcher)
                if denomination.startswith(cat):
                    positive_matches.append(n_matches)
                else:
                    negative_matches.append(n_matches)
        
        threshold = find_intersection(positive_matches, negative_matches)
        thresholds[filename] = threshold
        
        print(f"Se entreno la categoria: {comp_category}")

    with open(threshold_file, "w") as f:
        json.dump(thresholds, f, indent=4)
        
    print(f"Entrenamiento completado y guardado en {threshold_file}. Umbrales calculados para {len(thresholds)} billetes completos.")
    return thresholds

def predict_scoreboard(
        query_filepath: str,
        db_descriptors: dict[str, Descriptors],
        thresholds: dict[str, float],
        algorithm: Algorithm) -> None | dict[str, int]: 
    _, des_query = extract_features(query_filepath, algorithm)
    if des_query is None: return None

    matcher = None
    if (type(algorithm) is cv2.SIFT): 
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    scoreboard: dict[str, int] = defaultdict(int)

    for db_file, des_db in db_descriptors.items():
        filename = os.path.basename(db_file)
        category = get_category_from_db_filename(filename)
        denomination = get_denomination_from_category(category)
        
        if len(des_query) < 2 or len(des_db) < 2:
            continue

        n_good_matches = count_good_matches(des_query, des_db, matcher)
        threshold = thresholds.get(filename)
        
        if threshold is not None and n_good_matches >= threshold:
            scoreboard[denomination] += n_good_matches

    return scoreboard

def evaluate(
        test_set: DatasetFilepaths,
        db_descriptors: dict[str, Descriptors],
        thresholds: dict[str, float],
        algorithm: Algorithm):
    print("--- Evaluando el modelo con imágenes de prueba ---")
    correct = 0
    total = 0
    
    for true_cat, paths in test_set.items():
        for p in paths:
            scoreboard = predict_scoreboard(
                    p,
                    db_descriptors,
                    thresholds,
                    algorithm)

            predicted_cat = None
            if scoreboard is not None: 
                results = sorted(scoreboard.items(), key=lambda x: (x[1], x[0]), reverse=True)
                if results:
                    predicted_cat = results[0][0]
                
            is_correct = False
            if predicted_cat is not None:
                base_pred = predicted_cat.replace("Polimero", "")
                is_correct = (base_pred == true_cat)
                
            if is_correct: correct += 1
            total += 1
            
            print(f"Imagen: {os.path.basename(p)} | Real: {true_cat} | Predicción: {predicted_cat} | {'CORRECTO' if is_correct else 'INCORRECTO'}")
            
    accuracy = correct / total if total > 0 else 0
    print("-" * 40)
    print(f"Precisión Global (Accuracy): {accuracy * 100:.2f}% ({correct}/{total})")
    print("-" * 40)

# -------------------
# Ejecución principal

dataset = load_dataset(test_images_dir)
train_set, test_set = split_dataset(dataset, train_ratio=0.75)

print("Imágenes de entrenamiento por categoría:", {k: len(v) for k, v in train_set.items()})
print("Imágenes de prueba por categoría:", {k: len(v) for k, v in test_set.items()})

print("\n\n--------------------------------------------------------")
print("--------------------------SIFT--------------------------")
print("--------------------------------------------------------\n\n")

# Se utiliza el algoritmo SIFT (Scale-Invariant Feature Transform). 
# A diferencia de ORB, donde forzamos un número fijo de características (500),
# SIFT detecta automáticamente los puntos más estables basados en umbrales de 
# contraste local. 
# Aunque SIFT tiene un costo computacional mayor que ORB, es altamente robusto a 
# cambios de escala y rotación.
sift = cv2.SIFT.create()
full_sift_database_dir = os.path.join(database_dir, sift_database_dir) 
db_features = get_db_features(png_dir, full_sift_database_dir, sift)
thresholds = train_thresholds(train_set, db_features, sift)
evaluate(test_set, db_features, thresholds, sift)

print("\n\n--------------------------------------------------------")
print("--------------------------ORB---------------------------")
print("--------------------------------------------------------\n\n")

# Se utiliza el algoritmo ORB (Oriented FAST and Rotated BRIEF), en esta primera
# etapa solo se indica el número de características a extraer cuando se procesen
# imagenes.
# Se decide utilizar 2500 características ya que originalmente se dividian las 
# imagenes en 5 regiones, y utilizabamos 500 caracteristicas para cada una.
# Se considera basado en pruebas que 2500 es un buen número, ya que menos de 
# 2500 no captura suficientes detalles, mientras que más de 2500 no aporta
# mejoras significativas y aumenta el costo computacional.
orb = cv2.ORB.create(nfeatures=2500)
full_orb_database_dir = os.path.join(database_dir, orb_database_dir) 
db_features = get_db_features(png_dir, full_orb_database_dir, orb)
thresholds = train_thresholds(train_set, db_features, orb)
evaluate(test_set, db_features, thresholds, orb)

# -------------------
