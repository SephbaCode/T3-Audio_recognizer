import os

from PIL import Image
from matplotlib import pyplot as plt


def resultados(result, display = True, direct = "output/comparativas", name= "Resultado"):

    def takeSecond(elem):
        return elem[1]

    _result = sorted(result, key=takeSecond)
    lista, _scores, _resultado = zip(*_result)


    if not lista[0]==None:

        fig, ax = plt.subplots()
        plt.barh(lista, _scores)
        ax.set_ylabel('Canciones')
        ax.set_xlabel('Puntuación')
        ax.set_title('Resultados del matching')
        fig.tight_layout()

        if display:

            plt.show()
            name_long = name + "_comparativa_resultados"
            filename = direct + '\\' + name_long + ".png"
            #plt.savefig(filename)


        else:
            #este caso no es para mostrar la figura, solo para almacenarla
            name = name + "_comparativa_resultados"
            filename = direct + '\\' + name + ".png"
            plt.savefig(filename)