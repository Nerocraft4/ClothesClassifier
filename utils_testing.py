import Kmeans as km
import random
import time
from utils_data import Plot3DCloud, visualize_retrieval
import matplotlib.pyplot as plt

def visualitzar_kmeans(image, n_clusters):
    """
    Funció que ens mostrarà la classificació en kmeans d'una imatge
    concreta del dataset

    Args:
        image: Imatge en rgb
        n_clusters: Nombre de clusters amb els que volem calcular el Kmeans

    Returns:
        La funció no retorna res com a tal, però ens mostrarà un plot
        amb les diferents classes i els seus centroides
    """

    #Carraguem les dades
    data = image / 255.0 # use 0...1 scale, nomes ser
    # data = data.reshape(80*60, 3)

    #Cridem al Kmeans
    Km = km.KMeans(data, n_clusters)
    Km._init_centroids()
    Km.get_labels()
    Km.fit()

    plt.scatter(Km.X[:, 0], Km.X[:, 1], c=Km.labels, s=1, cmap='plasma_r');
    plt.scatter(Km.centroids[:,0] , Km.centroids[:,1] , s = 5, color = 'k')
    plt.show(block=True)
def plot3d (image, n_clusters):
    """
    Funció que ens mostrarà la classificació en kmeans d'una imatge
    concreta en un plot en 3D

    Args:
        dataset_img: Dataset de les imatges
        n_clusters: Nombre de clusters amb els que volem calcular el Kmeans

    Returns:
        La funció no retorna res, però ens mostrarà un plot 3D amb les
        diferents classes.
    """
    data = image
    Km = km.KMeans(data, n_clusters)
    Km._init_centroids()
    Km.get_labels()
    Km.fit()
    Plot3DCloud(Km, 1,1,1)
def queryColor(dataset_img, colors, question, n, viz = 0):
    """
    Funció que ens mostrarà les n imatges corresponents a un color
    entrat com a pregunta.

    Args:
        dataset_img: Dataset de les imatges
        colors: Llista on tenim els colors de les imatges un cop aplicat el Kmeans
        question: Valor de la pregunta, es a dir el color
        n: Nombre d'imatges que es volen mostrar
        viz: argument per si volem mostrar o no les imatges 0 per defecte(si) , 1(no)

    Returns:
        Retorna una llista de indexs els quals compleixen
        la condició de la pregunta
    """
    idx = []

    for i in range(len(colors)):
        if (question in colors[i]):
            idx.append(i)
            perct = colors[i].count(question)
    #Controlar q la n entrada no sigui mes gran q les imatges disponibles
    if(n > len(idx)):
        print("\nS'han demanat mes imatges de les que tenim disponibles, sortint...")
        return -1
    else:
        if(viz != 0):
            random.shuffle(idx)
            images_list = dataset_img[idx]
            visualize_retrieval(images_list, n)
        return idx
def queryLabels(dataset_img, labels, question, n, viz = 0):
    """
    Funció que ens mostrarà les n imatges corresponents a una classe
    entrat com a pregunta.

    Args:
        dataset_img: Dataset de les imatges
        labels: Llista on tenim les classes de les imatges un cop aplicat el KNN
        question: Valor de la pregunta, es a dir la classe
        n: Nombre d'imatges que es volen mostrar
        viz: argument per si volem mostrar o no les imatges 0 per defecte(si) , 1(no)

    Returns:
        Retorna una llista de indexs els quals compleixen
        la condició de la pregunta
    """
    idx = []
    for i in range(len(labels)):
        if (question in labels[i]):

            idx.append(i)

    #Controlar q la n entrada no sigui mes gran q les imatges disponibles
    if(n > len(idx)):
        print("\nS'han demanat mes imatges de les que tenim disponibles, sortint...")
        return -1
    else:
        if(viz != 0):
            random.shuffle(idx)
            images_list = dataset_img[idx]
            visualize_retrieval(images_list, n)
        return idx
def queryComplete(dataset_img, labels, colors, question, n):
    """
    Funció que ens mostrarà les n imatges corresponents a una classe i color
    entrat com a pregunta.

    Args:
        dataset_img: Dataset de les imatges
        labels: Llista on tenim les classes de les imatges un cop aplicat el KNN
        colors: Llista on tenim els colors de les imatges un cop aplicat el Kmeans
        question: Llista amb els valors de la pregunta primer element es el color, i el segon la classe
        n: Nombre d'imatges que es volen mostrar

    Returns:
        No retorna res com a tal, pero si que mostra les imatges que
        compleixen les condicions
    """
    #primer element de question es el color, i el segon la classe

    idx = queryLabels(dataset_img, labels, question[1], n, viz = 0)
    if(idx == -1):
        return
    dataset = dataset_img[idx]
    colorsn = colors[idx]

    idxn = queryColor(dataset, colorsn, question[0], n, viz = 0)
    if(idxn == -1):
        return
    images_list = dataset[idxn]

    visualize_retrieval(images_list, n)
def test_class_accuracy(ground_truth, predicted_labels):
    """
    Funció que calcula l'accuracy d'un conjunt de prediccions de
    classe, comparant-les amb les seves respectives ground_truth.

    Args:
        ground_truth: Llista de longitud l>0 que conté el ground_truth
        predicted_labels: Llista de longitud l>0 que conté les prediccions

    Returns:
        acc: el valor, en tant per cent, de l'encert de les prediccions
    """
    l=len(ground_truth)
    if l!=len(predicted_labels):
        print("Vectors have different lengths")
        return
    e = 0 #misprediction count
    for i in range(l):
        if ground_truth[i]!=predicted_labels[i]:
            e+=1
    acc=(l-e)/l*100
    return acc
def test_color_accuracy(ground_truth, predicted_labels):
    """
    Funció que calcula l'accuracy d'un conjunt de prediccions de
    color, comparant-les amb les seves respectives ground_truth.

    Args:
        ground_truth: Llista de longitud l>0 que conté el ground_truth
        predicted_labels: Llista de longitud l>0 que conté les prediccions

    Returns:
        acc: el valor, en tant per cent, de l'encert de les prediccions
    """
    l=len(ground_truth)
    if l!=len(predicted_labels):
        print("Vectors have different lengths")
        return
    tacc=0
    for i in range(l):
        truth = ground_truth[i]
        labels = predicted_labels[i]
        diff = len(list(filter(("White").__ne__,set(labels) - set(truth))))
        mxlen = max(len(labels),len(truth))
        if(mxlen==0): mxlen=1
        acc = (1-diff/mxlen)*100
        tacc += acc
    return round(tacc/l,2)
def test_convergence_speed(dataset_img, init_option, k):
    """
    Funció que calcula el temps mig de covergència del KMeans
    per a un conjunt d'imatges, amb un mètode d'init concret.

    Args:
        dataset_img: Dataset d'imatges en rgb
        init_option: Mode d'inicialització (first, random, custom o custom2)
        k: nombre de centroides amb que treballarà el KMeans

    Returns:
        mean_time: el temps mig de convergència de les imatges
    """
    start_time = time.time()
    for photo in dataset_img:
        model = km.KMeans(photo, k, options={'km_init':init_option})
        model.fit()
    mean_time = (time.time()-start_time)/len(dataset_img)
    return mean_time

def find_K_distribution(predicted_labels, max_K):
    """
    Funció que estudia la distribució de la best_k per a
    un conjunt d'imatges.

    Args:
        predicted_labels: Labels de color predits
        max_K: nombre màxim de centroides (i labels) del conjunt

    Returns:
        distribution: distribució en tant per u de la best_k.
    """
    histogram = [0 for i in range(max_K)]
    for labels in predicted_labels:
        histogram[len(labels)-1]+=1
    distribution = [round(elem/len(predicted_labels),4) for elem in histogram]
    return distribution
