# Aguirre_Jorge_P2.py
# Ejercicio: Web Scraping y Agrupación de Datos No Supervisado
# Fuente: https://books.toscrape.com/

import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
from urllib.parse import urljoin

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/"
MAX_REGISTROS = 120

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def limpiar_precio(texto):
    """
    Convierte un precio tipo '£51.77' o 'Â£51.77' a número decimal.
    """
    texto = str(texto)
    texto = texto.replace("£", "")
    texto = texto.replace("Â", "")
    texto = texto.replace(" ", "")
    texto = texto.strip()
    return float(texto)


def extraer_stock(texto):
    """Extrae el número de unidades disponibles desde el texto de disponibilidad."""
    match = re.search(r"\d+", texto)
    if match:
        return int(match.group())
    return 0


def obtener_sopa(url):
    """Descarga una página y devuelve el objeto BeautifulSoup."""
    respuesta = requests.get(url, headers=headers, timeout=15)
    respuesta.raise_for_status()
    return BeautifulSoup(respuesta.text, "html.parser")


def extraer_categoria_detalle(url_detalle):
    """Entra a la página individual del libro y extrae la categoría desde el breadcrumb."""
    sopa = obtener_sopa(url_detalle)
    breadcrumb = sopa.select("ul.breadcrumb li a")

    # En books.toscrape, la categoría suele estar en el tercer enlace del breadcrumb.
    if len(breadcrumb) >= 3:
        return breadcrumb[2].get_text(strip=True)
    return "Sin categoría"


def hacer_scraping(max_registros=MAX_REGISTROS):
    """Recorre las páginas del catálogo y obtiene al menos 100 registros."""
    registros = []
    pagina = 1

    while len(registros) < max_registros:
        url_pagina = f"{CATALOGUE_URL}page-{pagina}.html"
        print(f"Extrayendo página {pagina}: {url_pagina}")

        sopa = obtener_sopa(url_pagina)
        libros = sopa.select("article.product_pod")

        if not libros:
            break

        for libro in libros:
            if len(registros) >= max_registros:
                break

            titulo = libro.h3.a["title"]
            precio = limpiar_precio(libro.select_one("p.price_color").get_text(strip=True))
            disponibilidad_texto = libro.select_one("p.instock.availability").get_text(" ", strip=True)
            stock = extraer_stock(disponibilidad_texto)

            clases_rating = libro.select_one("p.star-rating").get("class", [])
            rating_palabra = [c for c in clases_rating if c != "star-rating"][0]
            rating = rating_map.get(rating_palabra, 0)

            url_relativa = libro.h3.a["href"]
            url_detalle = urljoin(url_pagina, url_relativa)
            categoria = extraer_categoria_detalle(url_detalle)

            registros.append({
                "titulo": titulo,
                "categoria": categoria,
                "precio": precio,
                "rating": rating,
                "stock": stock,
                "disponibilidad": disponibilidad_texto,
                "url": url_detalle
            })

            time.sleep(0.1)

        pagina += 1
        time.sleep(0.3)

    return pd.DataFrame(registros)


# =========================
# PARTE 1: WEB SCRAPING
# =========================

df = hacer_scraping(MAX_REGISTROS)
print("\nRegistros extraídos:", len(df))
print(df.head())

df.to_csv("dataset_libros_scraping.csv", index=False, encoding="utf-8-sig")
print("Dataset guardado como dataset_libros_scraping.csv")


# =========================
# PARTE 2: LIMPIEZA Y ANÁLISIS
# =========================

# Eliminación de duplicados y nulos importantes
df = df.drop_duplicates(subset=["titulo"])
df = df.dropna(subset=["titulo", "categoria", "precio", "rating", "stock"])

# Conversión de tipos
df["precio"] = df["precio"].astype(float)
df["rating"] = df["rating"].astype(int)
df["stock"] = df["stock"].astype(int)

# Tratamiento de texto
df["titulo_limpio"] = df["titulo"].str.lower().str.strip()
df["categoria_limpia"] = df["categoria"].str.lower().str.strip()
df["longitud_titulo"] = df["titulo_limpio"].str.len()

print("\n===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

# Guardar dataset limpio
df.to_csv("dataset_libros_limpio.csv", index=False, encoding="utf-8-sig")
print("Dataset limpio guardado como dataset_libros_limpio.csv")


# =========================
# PARTE 3: APRENDIZAJE NO SUPERVISADO
# =========================

features_numericas = ["precio", "rating", "stock", "longitud_titulo"]
features_categoricas = ["categoria_limpia"]

preprocesador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_categoricas)
    ]
)

kmeans = Pipeline(steps=[
    ("preprocesador", preprocesador),
    ("modelo", KMeans(n_clusters=4, random_state=42, n_init=10))
])

df["cluster"] = kmeans.fit_predict(df[features_numericas + features_categoricas])

print("\n===== CANTIDAD POR CLUSTER =====")
print(df["cluster"].value_counts().sort_index())

print("\n===== RESUMEN POR CLUSTER =====")
resumen_clusters = df.groupby("cluster")[["precio", "rating", "stock", "longitud_titulo"]].mean()
print(resumen_clusters)

# Categorías más comunes por cluster
print("\n===== CATEGORÍAS MÁS COMUNES POR CLUSTER =====")
for cluster in sorted(df["cluster"].unique()):
    print(f"\nCluster {cluster}")
    print(df[df["cluster"] == cluster]["categoria"].value_counts().head(5))

# Guardar dataset final con clusters
df.to_csv("dataset_libros_con_clusters.csv", index=False, encoding="utf-8-sig")
print("Dataset final guardado como dataset_libros_con_clusters.csv")


# =========================
# PARTE 5: VISUALIZACIONES
# =========================

# Transformar datos para PCA
X_transformado = kmeans.named_steps["preprocesador"].transform(df[features_numericas + features_categoricas])

# Si sale matriz dispersa, convertir a arreglo normal
if hasattr(X_transformado, "toarray"):
    X_transformado = X_transformado.toarray()

pca = PCA(n_components=2, random_state=42)
componentes = pca.fit_transform(X_transformado)
df["pca_1"] = componentes[:, 0]
df["pca_2"] = componentes[:, 1]

# 1. Scatter Plot por cluster
plt.figure(figsize=(8, 6))
plt.scatter(df["pca_1"], df["pca_2"], c=df["cluster"])
plt.title("Scatter Plot de Libros Agrupados por Cluster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()

# 2. Barras: cantidad de libros por cluster
plt.figure(figsize=(8, 5))
df["cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Cantidad de Libros por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de libros")
plt.show()

# 3. Heatmap del promedio de variables por cluster
plt.figure(figsize=(8, 5))
sns.heatmap(resumen_clusters, annot=True, cmap="Blues")
plt.title("Promedio de Variables por Cluster")
plt.show()

# 4. Distribución de precios por cluster
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="cluster", y="precio")
plt.title("Distribución de Precios por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Precio")
plt.show()


# =========================
# PARTE 4: INTERPRETACIÓN
# =========================

print("\n===== INTERPRETACIÓN BREVE =====")
print("1. Se formaron 4 grupos usando K-Means.")
print("2. Los clusters se diferencian principalmente por precio, rating, stock, longitud del título y categoría.")
print("3. Algunos grupos pueden concentrar libros más caros, mejor calificados o con mayor disponibilidad.")
print("4. Este análisis serviría para segmentar productos, detectar patrones de precios y organizar recomendaciones.")
