from __future__ import print_function

import os

import librosa
from matplotlib import pyplot as plt, ticker as plticker, colors as colors, cm, patches
import librosa.display
from scipy.signal import argrelextrema
import numpy
import pandas as pd


def mostrar_maximos_locales(Xdb, grid, percentil):
    """
    Máximos locales del spectograma
    """
    
    # Limites del grid sobre el area total
    xlins = numpy.linspace(0, Xdb.shape[1], num = grid[1]+1, dtype=numpy.int32)
    ylins = numpy.linspace(0, Xdb.shape[0], num = grid[0]+1, dtype=numpy.int32)
    
    # Cordenadas x, y de los maximos de cada area local
    x = []
    y = []
    
    for i in range(grid[0]):
        for j in range(grid[1]):

            # Area local a inspeccionar
#             print(ylins[i],ylins[i+1], xlins[j],xlins[j+1])
            Xlocal = Xdb[ylins[i]:ylins[i+1], xlins[j]:xlins[j+1]]

            #Comprobar que cumple con el percentil
            perc = numpy.percentile(Xdb.flatten(), percentil)
       
            if Xlocal.max() > perc:
                # Corrdenada del maximo del area local seleccionada
                coord_max = numpy.where(Xlocal == Xlocal.max()) 
                yij, xij  = coord_max[0][0]+ylins[i], coord_max[1][0]+xlins[j]
                x.append(xij)
                y.append(yij)

    return x, y


def distancia(p0, p1): 
    import math
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) 

def desfase_tiempo(p0, p1): 
    import math
    return p1[0] - p0[0]

def desfase_frecuencia(p0, p1): 
    import math
    return abs(p0[1] - p1[1])


def matriz_correlacion(max_locales, display = True):
    """
    Correlaciones en distancia, tiempo y frecuencia
    """
    
    cm = numpy.zeros(shape=(len(max_locales[1]), len(max_locales[1])))
    ct = numpy.zeros(shape=(len(max_locales[1]), len(max_locales[1])))
    cf = numpy.zeros(shape=(len(max_locales[1]), len(max_locales[1])))

    for i in range(len(max_locales[1])):
        for j in range(len(max_locales[1])):

            cm[i][j] = distancia(max_locales.T[i], max_locales.T[j])
            ct[i][j] = desfase_tiempo(max_locales.T[i], max_locales.T[j])
            cf[i][j] = desfase_frecuencia(max_locales.T[i], max_locales.T[j])
            
    if display:
        f = plt.figure(figsize=(10, 10))
        plt.matshow(cm, fignum=f.number)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Matriz de Correlación', fontsize=16)
        
    return cm, ct, cf


def seleccion_caracteristicas(cm, ct, cf, t_max, t_min, display=True, f_max=300, f_min=100):
    """
    Selección de caracteristicas
    """
    cc = numpy.where(numpy.logical_and(
                        numpy.logical_and(ct < t_max, ct > t_min),
                        numpy.logical_and(cf < f_max, cf > f_min)))
    
    if display:
        plt.plot(cc[0], cc[1], 'o', c='r')
    
    return cc


def crear_datos(Xdb, max_locales, cc, ut):
    """
    Xdb: Espectograma
    max_locales: maximos locales seleccionados entre todo el espectograma.
    cc: Array de localizaciones de caracteristicas buenas
    """
    
    f0=[]
    f1=[]
    dist=[]
    time=[]
    freq=[]
    X0=[]
    X1=[]
    Y0=[]
    Y1=[]

    for i in range(len(cc[0])):

        x0 = round(max_locales[0][cc[0][i]])
        y0 = round(max_locales[1][cc[0][i]])

        x1 = round(max_locales[0][cc[1][i]])
        y1 = round(max_locales[1][cc[1][i]])

        X0.append(x0*ut)
        X1.append(x1*ut)
        Y0.append(y0)
        Y1.append(y1)

        f0.append(round(Xdb[y0, x0]))
        f1.append(round(Xdb[y1, x1]))

        dist.append(round(distancia((x0, y0), (x1, y1))))

        time.append(round(abs(x1-x0)*ut))
        freq.append(round(abs(y1-y0)))
        
    d = {'f0': Y0, 'f1': Y1, 't0': X0, 't1': X1, 'dB0': f0, 'dB1': f1, 'dist': dist, 'utime': time, 'ufreq': freq}
    df = pd.DataFrame(data=d)
    ddf = df.drop_duplicates()
    


    return ddf


def crear_huella_digital(nombrearchivo, nombre="Default", display=True, test=False, n_fft=2048, hop_length=512,
                         percentil=90,
                         tmax=3, tmin=1, f_max=500, f_min=0, delim_freq=5):
    """
    Genera la huella digital de un archivo de audio y guarda el espectrograma con máximos locales.
    """
    # Cargar el archivo de audio
    x, sr = librosa.load(nombrearchivo)

    # Incremento unidad de tiempo
    ut = float(hop_length) / sr

    # Transformada de Fourier a corto plazo (STFT)
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    Xdb = librosa.amplitude_to_db(abs(X))

    # Calcular máximos locales
    tiempo_total = Xdb.shape[1] * ut
    grid_t = int(tiempo_total / tmin) if not test else int(tiempo_total / tmin) * 4
    grid = (delim_freq, grid_t)
    xi, yi = mostrar_maximos_locales(Xdb, grid, percentil)
    Pi_max = np.array([xi, yi])

    # Procesar características
    cm, ct, cf = matriz_correlacion(Pi_max, display=False)
    _tmax = round(tmax / ut)
    _tmin = round(tmin / ut)
    cc = seleccion_caracteristicas(cm, ct, cf, _tmax, _tmin, display=False, f_max=f_max, f_min=f_min)
    df = crear_datos(Xdb, Pi_max, cc, ut)

    # Especificar directorio de salida
    if display:
        direct = "output/spects"
    else:
        direct = "data/patrones/specshow"

    os.makedirs(direct, exist_ok=True)  # Crear el directorio si no existe
    file = f"{direct}/{nombre}.png"

    # Generar espectrograma
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(Xdb, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.plot(Pi_max[0] * hop_length / sr, Pi_max[1] * sr / (2 * len(Xdb)), 'o', c='r')  # Máximos locales
    plt.colorbar(format='%+2.0f dB')
    plt.title("Espectrograma con máximos locales - " + nombre)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.tight_layout()

    # Mostrar y guardar
    if display:
        # plt.savefig(file, format='png', dpi=300)
        plt.show()

    # Guardar imagen
    # plt.savefig(file, format='png', dpi=300)
    plt.close()  # Cerrar la figura después de guardarla

    return df, Xdb, Pi_max

def coincidencia_v1(fingerAi, finger_i):
    """
    Coincidencia entre dos huellas
    """
    it = 0
    ta = 0
    tb = 0
    
    coincidencias = fingerAi[numpy.logical_and(numpy.logical_and(fingerAi["f0"]==finger_i["f0"], 
                                                 fingerAi["f1"]==finger_i["f1"]), 
                                                 fingerAi["utime"]==finger_i["utime"])]
    
    if len(coincidencias)>0:
        for i in range(len(coincidencias)):

            ta=coincidencias["t0"].iloc[i]
            tb=finger_i["t0"]
            it = it+1

        return True, it, ta, tb

    else:
        return False, it, ta, tb


def coincidencia_total(fingerA, fingerBt_i, name, display=True, min_match=5, tmax = 60, direct = "data/test/fp"):

    # Conjuntos de match correlacionado
    tA = []
    tB = []

    for i in range(len(fingerBt_i)):
        resu, _, ta, tb = coincidencia_v1(fingerA, fingerBt_i.iloc[i])

        if resu:
            tA.append(ta)
            tB.append(tb)

    if len(tA)>0:
        fig, axs = plt.subplots(2, 1, figsize=(8,6))

        axs[0].set_title('Diagonal presente')
        axs[0].plot(tA, tB, 'o')
        axs[0].set_xlim([0, tmax])
        axs[0].set_ylim([0, max(tB)+1])
        axs[0].set_xlabel('Tiempo de archivo de sonido de la base de datos')
        axs[0].set_ylabel('Tiempo de archivo de sonido de la muestra')
        axs[0].grid(True)

        axs[1].set_title('Señales coincidentes')
        n = axs[1].hist(tA, numpy.arange(0, tmax, 5))
        axs[1].set_xlim([0, tmax])
        # axs[1].set_ylim([0, n+10])
        axs[1].set_xlabel('Tiempo de archivo de sonido de la base de datos')
        axs[1].set_ylabel('Tiempo de archivo de sonido de la muestra')

        if display:
            fig.tight_layout()
            # plt.show()
        else:
            fig.tight_layout()
            file_img = direct + "/" + name + ".png"
            plt.savefig(file_img)

        element_max = numpy.where(n[0]==max(n[0]))[0][0]

        if element_max==n[0].shape[0]-1:
            tot = element_max
        else:
            if element_max==0:
                tot = n[0][element_max] + n[0][element_max+1]
            else:
                tot = n[0][element_max] + n[0][element_max+1]+ n[0][element_max-1]

        # ratio = tot/sum(n[0])
        if tot >= min_match:
            ratio = tot
        else:
            ratio = 0.0

    else:
        ratio = 0.0

    return ratio, tA, tB


import numpy as np  # Asegúrate de que numpy esté importado


def coincidencia_total_sin_graficas(fingerA, fingerBt_i, min_match=5, tmax=60):
    """
    Calcula coincidencias entre una muestra de audio y una base de datos de huellas digitales,
    sin generar gráficas en el proceso.
    """

    # Conjuntos de match correlacionado
    tA, tB = [], []

    # Iterar sobre las huellas digitales en la base de datos
    for i in range(len(fingerBt_i)):
        resu, _, ta, tb = coincidencia_v1(fingerA, fingerBt_i.iloc[i])

        if resu:
            tA.append(ta)
            tB.append(tb)

    # Si hay coincidencias, calcular el histograma y el ratio
    if tA:
        # Crear histograma de coincidencias
        n, bins = np.histogram(tA, bins=np.arange(0, tmax + 5, 5))

        # Calcular el máximo elemento del histograma
        element_max = np.argmax(n)

        # Sumar coincidencias relevantes
        tot = n[element_max]
        if element_max > 0:
            tot += n[element_max - 1]
        if element_max < len(n) - 1:
            tot += n[element_max + 1]

        # Calcular el ratio de coincidencias
        ratio = tot if tot >= min_match else 0.0
    else:
        ratio = 0.0

    return ratio, tA, tB

