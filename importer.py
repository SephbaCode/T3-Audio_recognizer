import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.io import wavfile

def seleccionar_archivo_wav():
    try:
        # Crear la ventana principal de Tkinter (oculta)
        root = tk.Tk()
        root.withdraw()

        # Abrir un cuadro de diálogo para seleccionar un archivo WAV
        archivo_wav = filedialog.askopenfilename(
            title="Seleccionar archivo WAV",
            filetypes=[("Archivos WAV", "*.wav")]
        )

        # Leer el archivo WAV; sample rate = frecuencia de muestreo, audio = señal de audio
        sample_rate, audio = wavfile.read(archivo_wav)

        # Si el audio tiene dos canales (estéreo), convertirlo a mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        nombre = archivo_wav.split("/")[-1]
        nombre = nombre.split(".")[0]

        ruta = archivo_wav

    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return None, None, None

    return nombre, sample_rate, audio, ruta



