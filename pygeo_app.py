import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from dtaidistance import dtw
from scipy.signal import correlate

# Título de la aplicación
st.title("PyGeo: Correlación de Pozos con Múltiples Técnicas")

# Subir archivo ZIP que contenga múltiples archivos LAS
uploaded_file = st.file_uploader("Sube un archivo ZIP que contenga archivos LAS", type="zip")

if uploaded_file is not None:
    # Descomprimir el archivo ZIP
    with zipfile.ZipFile(uploaded_file, "r") as z:
        las_files = [z.open(name) for name in z.namelist() if name.endswith(".las")]
        st.write(f"Archivos .las encontrados: {[name for name in z.namelist() if name.endswith('.las')]}")

    # Leer y almacenar los archivos LAS
    pozos_data = {}
    for las_file in las_files:
        las = lasio.read(las_file)
        df_las = las.df()  # Convertir a DataFrame
        pozo_nombre = las.well.WELL.value  # Nombre del pozo
        pozos_data[pozo_nombre] = df_las

    st.write("Datos cargados con éxito!")

    # Mostrar resumen de los datos cargados
    for pozo, data in pozos_data.items():
        st.subheader(f"Pozo: {pozo}")
        st.write(data.describe())  # Mostrar resumen estadístico de los datos del pozo

    # Seleccionar los pozos para correlacionar
    st.subheader("Correlación de Pozos")
    pozo_nombres = list(pozos_data.keys())
    pozo_1 = st.selectbox("Selecciona el primer pozo para correlacionar", pozo_nombres)
    pozo_2 = st.selectbox("Selecciona el segundo pozo para correlacionar", pozo_nombres)

    # Seleccionar las curvas para correlacionar
    curvas_disponibles = pozos_data[pozo_1].columns
    curva_seleccionada_1 = st.selectbox(f"Selecciona la curva para {pozo_1}", curvas_disponibles)
    curva_seleccionada_2 = st.selectbox(f"Selecciona la curva para {pozo_2}", curvas_disponibles)

    # Extraer los datos de las curvas seleccionadas
    data_1 = pozos_data[pozo_1][curva_seleccionada_1].dropna().values
    data_2 = pozos_data[pozo_2][curva_seleccionada_2].dropna().values

    # Elegir la técnica de correlación
    st.subheader("Selecciona la técnica de correlación")
    tecnica_seleccionada = st.selectbox("Técnica de correlación", 
                                        ["Cross-Correlation", "DTW", "CCA", "Información Mutua", "Clustering KMeans"])

    # Funciones de correlación
    def correlacion_cruzada(curva1, curva2):
        correlacion = correlate(curva1, curva2)
        return correlacion

    def correlacion_dtw(curva1, curva2):
        distancia = dtw.distance(curva1, curva2)
        path = dtw.warping_path(curva1, curva2)
        return distancia, path

    def correlacion_cca(curvas1, curvas2):
        cca = CCA(n_components=1)
        cca.fit(curvas1.reshape(-1, 1), curvas2.reshape(-1, 1))
        correlacion1, correlacion2 = cca.transform(curvas1.reshape(-1, 1), curvas2.reshape(-1, 1))
        return correlacion1, correlacion2

    def informacion_mutua(curva1, curva2):
        return mutual_info_score(curva1.astype(int), curva2.astype(int))

    def clustering_kmeans(data_1, data_2):
        combined_data = np.vstack((data_1, data_2)).T
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        return labels

    # Ejecutar la técnica seleccionada
    if tecnica_seleccionada == "Cross-Correlation":
        resultado = correlacion_cruzada(data_1, data_2)
        st.write(f"Resultado de la correlación cruzada: {resultado}")
        fig, ax = plt.subplots()
        ax.plot(resultado)
        ax.set_title("Correlación Cruzada")
        st.pyplot(fig)

    elif tecnica_seleccionada == "DTW":
        distancia, path = correlacion_dtw(data_1, data_2)
        st.write(f"Distancia DTW entre {pozo_1} y {pozo_2}: {distancia}")
        fig, ax = plt.subplots()
        ax.plot(data_1, label=f"{pozo_1} - {curva_seleccionada_1}", color="blue")
        ax.plot(data_2, label=f"{pozo_2} - {curva_seleccionada_2}", color="red")
        for (i, j) in path:
            ax.plot([i, j], [data_1[i], data_2[j]], color="grey", linestyle="--")
        ax.legend()
        ax.set_title("Correlación DTW")
        st.pyplot(fig)

    elif tecnica_seleccionada == "CCA":
        correlacion1, correlacion2 = correlacion_cca(data_1, data_2)
        st.write(f"Correlación Canónica entre {pozo_1} y {pozo_2}")
        fig, ax = plt.subplots()
        ax.plot(correlacion1, label=f"{pozo_1} - CCA", color="blue")
        ax.plot(correlacion2, label=f"{pozo_2} - CCA", color="red")
        ax.legend()
        ax.set_title("Correlación Canónica (CCA)")
        st.pyplot(fig)

    elif tecnica_seleccionada == "Información Mutua":
        info_mutua = informacion_mutua(data_1, data_2)
        st.write(f"Información mutua entre {pozo_1} y {pozo_2}: {info_mutua}")

    elif tecnica_seleccionada == "Clustering KMeans":
        labels = clustering_kmeans(data_1, data_2)
        st.write(f"Clustering KMeans entre {pozo_1} y {pozo_2}")
        fig, ax = plt.subplots()
        ax.scatter(data_1, data_2, c=labels)
        ax.set_title("Clustering KMeans")
        st.pyplot(fig)

else:
    st.info("Por favor, sube un archivo ZIP con archivos LAS para comenzar.")
