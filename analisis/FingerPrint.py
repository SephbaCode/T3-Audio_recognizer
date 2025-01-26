import os

import numpy
import pandas as pd

from DB.BaseDatos import add_registro, registro
from analisis.Shazam import crear_huella_digital, coincidencia_total, coincidencia_total_sin_graficas


def crear_fingerprint(filename, filename_fp, name, nueva=False):

    # si e esta agragando la cancion a la base de datos se crea la huella digital y se le asigna un nombre
    if not nueva:
        dfC, _, _ = crear_huella_digital(filename, test=False, n_fft=2048, hop_length=512, percentil=90,
                tmax=3, tmin=1, f_max=500, f_min=0, delim_freq = 5, nombre=name)

        # se guarda la huella digital en un archivo csv y se añade el nombre a la base de datos
        dfC.to_csv(filename_fp)
        print(".... Fingerprint almacenada en: ", filename_fp)
        add_registro(name)
        print(".... Fingerprint añadido en Base de Datos: ", name)

    # si se esta realizando una busqueda de la cancion se crea la huella digital con display esto permite que se muestre el espectograma
    else:
        dfC, _, _ = crear_huella_digital(filename, nombre=name, display=True, test=False, n_fft=2048, hop_length=512, percentil=90,
                tmax=3, tmin=1, f_max=500, f_min=0, delim_freq = 5)

    print(".... Fingerprint creada")

    # retorna una dataframe con la huella digital
    return dfC


def Busqueda_matching(dfC, min_match=5):

    # informacion sobre direcciones fingerprints disponibles a comparacion
    lista = registro()
    test_fp_filename = os.path.join("data", "patrones", "fingerprints")
    resultados = []

    # itera en todos los elementos de la lista
    for elemento in lista:
        archivo = test_fp_filename + "\\" + elemento + ".csv"
        print("Archivo analizado en este ciclo: ", archivo)
        df = pd.read_csv(archivo)

        # esta funcion retorna el ratio de coincidencia entre dos huellas digitales parar un elemento de la base de datos
        # ratio, _, _ = coincidencia_total(df, dfC, elemento, min_match=min_match)
        ratio, _, _ = coincidencia_total_sin_graficas(df, dfC, min_match=min_match)

        resultados.append(ratio)

    # si hay resultados y la suma de estos es mayor a 0
    if len(resultados) > 0 and sum(resultados) > 0.0:
        na_result = numpy.array(resultados)                                 # se convierte a un array de numpy
        id_Result_final = numpy.where(na_result == max(na_result))[0][0]    # se obtiene el indice del resultado con mayor coincidencia

        _match = lista[id_Result_final]                         # se obtiene el nombre de la cancion con mayor coincidencia en base al indice
        _coincidencias = resultados[id_Result_final]            # se obtiene el ratio de coincidencia de la cancion con mayor coincidencia
        _score = round(_coincidencias / sum(resultados), 2)     # se obtiene el score de la cancion con mayor coincidencia

        _scores_aux = resultados / sum(resultados)
        _scores = [round(i, 2) for i in _scores_aux]

        _resultados = [int(i) for i in resultados]

        _result = list(zip(lista, _scores, _resultados))

    else:
        _match, _scores, _resultados = "Sin Coincidencias", 0.0, [0]
        _score, _result = 0.0, 0.0

        _result = (_match, _scores, _resultados)

    return _match, _score, _result

