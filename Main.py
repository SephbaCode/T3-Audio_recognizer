import os

from analisis.Resultados import resultados
from importer import seleccionar_archivo_wav
from analisis.FingerPrint import crear_fingerprint, Busqueda_matching


def menu_principal():

    data_filename = os.path.join("data", "patrones", "files")
    fingerprint_filename = os.path.join("data", "patrones", "fingerprints")
    test_filename = os.path.join("data", "test")
    test_fp_filename = os.path.join("data", "test", "fp")

    print("NOTAS IMPORTANTES:\n "
          "     - Para usar su funcion de reconocimiento el programa\n"
          "       requiere un archivo wav existente en una ubicación\n"
          "       conocida parar proceder a analizar.\n"
          "     - El siguiente programa cuenta con una cantidad limitada\n"
          "       de pistas que puede reconocer, Consultar la lista de\n"
          "       canciones reconocibles.\n"
          "     - SE RECOMIENDA que el archivo sea de 10 segundos de duracion.\n"
          "     - SE SUGIERE usar las pistas grabades cortas de la carpeta data/test/\n")
    while True:
        print("\n******************************************************")
        print("***************** MENU PRINCIPAL *********************")
        print("******************************************************")
        print("1. Reconocer grabacion")
        print("2. Lista decanciones reconocibles")
        print("3. Salir")
        choice = input("Seleccione una opción: ")

        if choice == '1':
            print("Has seleccionado reconocimiento de grabacion wav.")
            print("Selecciona un archivo wav para reconocer.")

            nombre, sr, audio, ruta = seleccionar_archivo_wav()

            # direcciones de almacenamiento
            filename = ruta
            filename_fp = test_fp_filename + "\\" + nombre + ".csv"
            print("Nombre de la grabacion Seleccionada: " + nombre)
            print("Ruta en la que se almacenara la FingerPrint: " + filename_fp)

            # Generacion de la huella
            print("Generando fingerprint del audio grabado ...")

            dfB = crear_fingerprint(filename, filename_fp, nombre, nueva=True)

            item, score, result = Busqueda_matching(dfB, min_match=5)
            print("-------------------------------------------------------")
            print("Resultado: ")

            if not item == "Sin Coincidencias":

                print("Canción: ", item)
                print("Score: ", score)
                # print("General: ", result)
                print("-------------------------------------------------------")
                print("\n")

                # esta parte del codigo mostrara los resultados en varias grafica
                resultados(result, display=True, name=nombre)

        elif choice == '2':
            # lista de canciones reconocibles
            print("******************************************************")
            print("***************** Lista de Canciones *****************")
            print("******************************************************")

            #lectura del archivo BASE_DATOS.TXT EN LA CARPETA DATA
            with open("data/patrones/BASE_DATOS.txt", "r") as file:
                print(file.read())


        elif choice == '3':
            print("Saliendo del menú...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    menu_principal()