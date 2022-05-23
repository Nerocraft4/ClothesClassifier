import Kmeans as km
import random
import time
from utils_data import *
from utils_testing import *
import matplotlib.pyplot as plt

def Filtre(cerca):
    """
    Funció que convertirà la cerca entrada per l'usuari al format correcte, la primera lletra amb
    majúscules ,per la cerca posterior en els Query

    Args:
        cerca: paraula amb el corresponent valor de la pregunta

    Returns:
        Retorna la paraula amb el format correcte

    """
    cerca_min = cerca.lower()
    return cerca_min.capitalize()

def menu(dataset, labels, colors):
    print("Menú de selecció Query:\n")
    a = 1

    while (a == 1):

        print("\nSelecciona el nombre de la cerca:")
        print("1- Query Colors")
        print("2- Query Labels")
        print("3- Query Complete (Color and Labels)")
        print("4- Sortir")

        select = int(input("Selecció: "))

        if(select == 1):
            a = -1
            print("Has cridat al query de Colors")
            print("Llista colors disponibles:\n - Red\n - Orange\n - Brown\n - Yellow\n - Green\n - Blue\n - Purple\n - Pink\n - Black\n - Grey\n - White\n")

            cerca = input("Introdueix color a buscar: ")
            #Apliquem un filtre perque el format sempre sigui el bo
            cerca = Filtre(cerca)

            #Fer el control de cerca
            disp = ['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Black', 'Grey', 'White']
            if(cerca in disp):
                n = int(input("Introdueix el nombre d'imatges que vols veure: "))
                queryColor(dataset, colors, cerca ,n, 1)
            else:
                print("\nError d'entrada, color no disponible, sortint...")
                a =  1


        elif(select == 2):
            a = -1
            print("\nHas cridat al query de Labels")
            print("Llista classes disponibles:\n - Jeans\n - Heels\n - Handbags\n - Flip Flops\n - Shorts\n - Socks\n - Dresses\n - Shirts\n - Sandals")

            cerca = input("Introdueix la classe a buscar: ")
            #Apliquem un filtre perque el format sempre sigui el bo
            if (cerca != "Flip Flops"):
                cerca = Filtre(cerca)

            #Fer el control de cerca
            disp = ['Jeans', 'Heels', 'Handbags', 'Flip Flops', 'Shorts', 'Socks', 'Dresses', 'Shirts', 'Sandals']
            if(cerca in disp):
                n = int(input("Introdueix el nombre d'imatges que vols veure: "))
                queryLabels(dataset, labels, cerca ,n, 1)
            else:
                print("\nError d'entrada, classe no disponible, sortint...")
                a = 1


        elif(select == 3):
            a = -1
            print("\nHas cridat al query Complet")
            print("Llista classes disponibles i colors disponibles:\n"+
                  " - Red\t\t - Jeans\n"+
                  " - Orange\t - Heels\n"+
                  " - Brown\t - Handbags\n"+
                  " - Yellow\t - Flip Flops\n"+
                  " - Green\t - Shorts\n"+
                  " - Blue\t\t - Socks\n"+
                  " - Purple\t - Dresses\n"+
                  " - Pink\t\t - Shirts\n"+
                  " - Black\t - Sandals\n"+
                  " - Grey\n"+
                  " - White\n ")

            cercaC = input("Introdueix el color a buscar: ")
            cercaL = input("Introdueix la classe a buscar: ")

            #Apliquem un filtre perque el format sempre sigui el bo
            cercaL = Filtre(cercaL)
            cercaC = Filtre(cercaC)

            dispC =  ['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Black', 'Grey', 'White']
            dispL = ['Jeans', 'Heels', 'Handbags', 'Flip Flops', 'Shorts', 'Socks', 'Dresses', 'Shirts', 'Sandals']
            #Fer el control de cerca
            if(cercaL in dispL and cercaC in dispC):
                n = int(input("Introdueix el nombre d'imatges que vols veure: "))
                queryComplete(dataset, labels, colors, [cercaC, cercaL], n)
            else:
                print("\nError d'entrada, sortint...")
                a=1

        elif(select == 4):
            print("\nSortint...\n")
            return

        else:
            print("\nNombre no conegut o entrada fallida, selecciona un nombre correcte\n")
            a = 1

        if(a!=1):
            sortida = input("Vols fer una altre cerca S/n: ")

            if(sortida == "S"): a = 1
            else:
                print("\nSortint...")
                return

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
train_imgs, train_class_labels, train_color_labels, \
test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')
classes = list(set(list(train_class_labels) + list(test_class_labels)))
all_data = np.concatenate((train_imgs,test_imgs))
all_data_class_labels = np.concatenate((train_class_labels,test_class_labels))
all_data_color_labels = np.concatenate((train_color_labels,test_color_labels))

picks = get_picks(len(all_data),350)
s1 = get_sample(all_data,picks)
s2 = get_sample(all_data_class_labels,picks)
s3 = get_sample(all_data_color_labels,picks)
menu(s1,s2,s3)
