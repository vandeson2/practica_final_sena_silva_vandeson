import pandas as pd
import numpy as np
def media(list_datos: list):
    pass

def mediana(list_datos: list):
    pass

def percentil(list_datos: list):
    pass
def varianza(list_datos: list):
    pass
def desviacion(list_datos: list):
    pass
def iqr(list_datos: list):
    pass
if __name__ == "__main__":
    np.random.seed(42)
    edad = list(np.random.randint(28, 68, 100))
    salario = list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random(0, 30, 100))

    np.random.seed(42)

    df = pd.DataFrame({
        'edad': np.random.randint(28, 68, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random(0, 30, 100)
    })
    ## hacerlo con todas las funciones que se han creado
    print("Resultado pandas: ")
    print("---------------------------")
    print(df.describe())

    print("Resultado edad: ")
    print("---------------------------")
    print(media(edad))

    print("Resultado salario: ")
    print("---------------------------")
    print(media(salario))
    
    print("Resultado experiencia: ")
    print("---------------------------")
    print(media(experiencia))
