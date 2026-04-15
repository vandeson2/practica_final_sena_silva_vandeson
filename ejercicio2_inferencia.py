import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

def data_load(path: str):
    '''
    Carga el dataset

    parámetros: path -> Ruta del archivo CSV.
    return: DataFrame con los datos cargados. 
    '''
    df =  pd.read_csv(path)
    return df 

def clear_data(df: pd.DataFrame):
    '''
        Limpia el Dataset para el análisis

        Acciones realizadas:
        - Elimina columnas de id y de texto que no son útiles para el análisis.
        - Elimina variable de fecha 'last_review' con muchos nulos.
        - Reemplaza los valores nulos por 0 en.
    
    '''
    df_limpio = df.copy()
    columnas_eliminar = ['id', 'name', 'host_id', 'host_name', 'last_review']
    df_limpio = df_limpio.drop(columns=[col for col in columnas_eliminar if col in df_limpio.columns])

    if "reviews_per_month" in df_limpio.columns:
        df_limpio["reviews_per_month"]  = df_limpio["reviews_per_month"].fillna(0)

    return df_limpio

def construir_preprocesamiento(x: pd.DataFrame):
    '''
        Preprocesador para variables numéricas y categóricas.

        Parámetros:
        - x: DataFrame 

        Return:
        - preprocesador: ColumnTransforme configurado.
        - cols_num: lista de columnas numéricas.
        - cols_cat: lista de columnas categóricas.
    '''
    cols_num = x.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat)
        ]
    )

    return preprocesador, cols_num, cols_cat

def guardar_metricas_regresion(mae: float, rmse: float, r2:float, ruta_txt:str):
    '''
        Guarda las métricas de la regresión lineal en un archivo txt

        Parámetros:
        - mae: Mean Absolute Error.
        - rmse: Root Mean Squared Error.
        - r2: coeficiente de determinación.
        - ruta_txt: ruta del archivo de salida.
    '''
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("--- MÉTRICAS REGRESIÓN LINEAL ---\n\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

def obtener_nombres_variables(preprocesador: ColumnTransformer, cols_num: list, cols_cat: list):
    '''
        Recupera los nombres de las variables tras el preprocesamiento.

        Parámetros:
        - preprocesador: ColumnTransformer configurado.
        - cols_num: lista de columnas numéricas.
        - cols_cat: lista de columnas categóricas.

        Return:
        - Lista con los nombres de las características transformadas.
    '''
    nombres = list(cols_num)

    if cols_cat:
        encoder = preprocesador.named_transformers_["cat"]
        nombres_cat = encoder.get_feature_names_out(cols_cat).tolist()
        nombres.extend(nombres_cat)
    
    return nombres

def grafica_coeficientes(coeficientes: np.ndarray, nombre_variables: list, ruta_png: str):
    '''
        Genera un gráfico con los 10 coeficientes de mayor valor absoluto.

        Parámetros:
        - coeficientes: array de coeficientes del modelo lineal.
        - nombres_variables: nombre de las variables transformadas.
        - ruta_png: ruta de salida de la imagen.
    '''
    serie_coef = pd.Series(coeficientes, index=nombre_variables)
    top_10 = serie_coef.abs().sort_values(ascending=False).head(10).index
    serie_top = serie_coef.loc[top_10].sort_values()

    plt.figure(figsize=(10, 6))
    plt.barh(serie_top.index, serie_top.values)
    plt.title("Top 10 coeficientes con mayor peso absoluto")
    plt.xlabel("Coeficiente")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300, bbox_inches="tight")
    plt.close()

def grafica_residuos(y_real: pd.Series, y_pred: np.ndarray, ruta_png: str):
    '''
        Genera el gráfico de residuos.

        Parámetros:
        - y_real: valores reales.
        - y_pred: valores predichos.
        - ruta_png: ruta de salida de la imagen.

    '''
    residuos = y_real - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.title("Gráfico de residuos")
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300, bbox_inches="tight")
    plt.close()








def main():
    datos = "data/listings.csv"
    carpeta_salida = "output"

    os.makedirs(carpeta_salida, exist_ok=True)

    df_original = data_load(datos)
    df_limpio = clear_data(df_original) 


    #Definir x e y para regresion lineal
    x = df_limpio.drop(columns=["price"])
    y = df_limpio["price"]

    #Construir preprocesador
    preprocesador, cols_num, cols_cat = construir_preprocesamiento(x)

    #Train/ test para regresión lineal
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    ) 

    #modelo de regresión lineal
    modelo_lineal = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", LinearRegression()),
        ]
    )

    modelo_lineal.fit(x_train, y_train)
    y_pred = modelo_lineal.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    guardar_metricas_regresion(
        mae, rmse, r2, os.path.join(carpeta_salida, "ej2_metricas_regresion.txt"),
    )

    prep = modelo_lineal.named_steps["preprocesador"]
    model = modelo_lineal.named_steps["modelo"]

    nombres = obtener_nombres_variables(prep, cols_num, cols_cat)

    grafica_coeficientes(model.coef_, nombres, os.path.join(carpeta_salida, "ej2_coeficientes.png"))
    grafica_residuos(y_test, y_pred, os.path.join(carpeta_salida, "ej2_residuos.png"))

if __name__ == "__main__":
    main()