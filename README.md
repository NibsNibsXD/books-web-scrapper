# Informe breve del proyecto

## Web Scraping y agrupación de datos no supervisada

La prueba de hoy  consiste en obtener datos reales desde una página web, limpiarlos, analizarlos y aplicar un algoritmo de aprendizaje no supervisado para encontrar patrones entre los registros.

## 1. Fuente de datos

Se utilizó la página **Books to Scrape**, una web de prueba que contiene información de libros. Esta página permite extraer datos de forma sencilla usando `requests` y `BeautifulSoup`.

Los datos extraídos fueron principalmente:

- Título del libro
- Categoría
- Precio
- Rating
- Disponibilidad en stock
- URL del producto

Se extranjeron 120 registros

## 2. Proceso de web scraping

Primero, el programa recorrió varias páginas del catálogo de libros. En cada página se identificaron los elementos HTML donde se encontraba la información de cada libro.

Con `BeautifulSoup` se extrajeron los datos principales y luego se guardaron en un `DataFrame` de `pandas`. Finalmente, el dataset obtenido fue exportado a un archivo CSV para conservar los datos originales del scraping.

## 3. Limpieza y preparación de datos

Después del scraping, se realizó una limpieza básica del dataset. Se eliminaron valores nulos, se convirtió el precio a formato numérico y se transformaron variables como el rating y el stock para que pudieran ser utilizadas por el modelo.

También se codificaron algunas variables de texto, como la categoría, para que el algoritmo pudiera procesarlas correctamente.

## 4. Análisis exploratorio

Se utilizaron funciones como:

```python
.head()
.info()
.describe()
```

Estas funciones permitieron revisar las primeras filas, los tipos de datos, la cantidad de registros y algunas estadísticas generales del dataset.

## 5. Aprendizaje no supervisado

Para la parte de Machine Learning se utilizó el algoritmo **K-Means**, que permite agrupar datos según similitudes entre sus características.

El modelo agrupó los libros en diferentes clusters tomando en cuenta variables como precio, rating, stock, longitud del título y categoría.

## 6. Visualizaciones

Se generaron varios gráficos para interpretar mejor los resultados:

- Scatter plot de los libros agrupados por cluster
- Gráfico de barras con la cantidad de libros por cluster
- Heatmap con el promedio de variables por cluster
- Distribución de precios por cluster

Estas visualizaciones ayudaron a observar diferencias entre los grupos generados por el modelo.

## 7. Interpretación general

El algoritmo permitió encontrar grupos de libros con características similares. Algunos clusters pueden representar libros más caros, otros libros con mejores calificaciones o con mayor disponibilidad.

Este tipo de análisis puede ser útil para segmentar productos, organizar catálogos, estudiar precios o apoyar sistemas de recomendación.

## Conclusión

El proyecto cumple con el proceso completo de web scraping, limpieza de datos, análisis exploratorio, aplicación de clustering y visualización de resultados. A partir de datos reales, se logró aplicar aprendizaje no supervisado para encontrar patrones dentro del conjunto de libros extraídos.

## Link para Colab

https://colab.research.google.com/drive/1UpZ2SQm60BoVqJMe-O-Iw-scCk71nE2O#scrollTo=ab0597cf
