from random import randint
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import math

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Function used mainly to visualize long tasks. This function does not
    impact the functionality of our code in any way, we used it mainly
    to have something to look at while waiting in the terminal.

    The original source is the following:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    This is the original documentation for the function:
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str) █
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |'+bcolors.OKCYAN+f'{bar}'+bcolors.ENDC+f'| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def grayscalize(dataset):
    """
        turns a grayscale from a one dimensional vector rgb to a single
        hue value (from the HSL representation)
    """
    shape = dataset.shape
    l=shape[0]
    c=0
    for photo in range(shape[0]):
        c+=1
        printProgressBar(c + 1, l, prefix = 'Grayscalizing:', suffix = 'Complete', length = 50)
        for px in range(shape[1]):
            for py in range(shape[2]):
                item = dataset[photo][px][py]
                dataset[photo][px][py]=0.299*item[0]+0.587*item[1]+0.114*item[2]
    return dataset

def k_fold(dataset,k,q):
    """
        samples a dataset (a list) using the k-fold technique. It slices
        the dataset in k chunks with (almost) the same sizes, and divides
        it in train and test. Chunk number q (zero indiced) will be test,
        while the other chunks will be train.
    """
    l=len(dataset)
    if q>=k:
        raise ValueError("q no pot ser >= k")
    if k>1 and k<l and k<15:
        c=round(np.floor(l/k))
        if q==0: #Primer cas, k-fold agafa el primer set
            test = dataset[:c]
            train = dataset[c:]
        elif q==k-1:
            test = dataset[l-c:]
            train = dataset[:l-c]
        else:
            test = dataset[q*c:(q+1)*c]
            train1 = dataset[:q*c]
            train2 = dataset[(q+1)*c:]
            train = np.concatenate((train1,train2))
        return train, test

def get_picks(rng,size):
    """
        creates a list of indices of size up to 'size' and with integers
        in the interval [0,rng]. Uniqueness is *not* guaranteed.
    """
    picks = set([randint(0, rng-1) for k in range(size)])
    return list(picks)

def get_sample(dataset,picks):
    """
        samples a dataset (a list) from a list of indices, 'picks'
    """
    return np.array([dataset[pick] for pick in picks])

def get_random_sample(dataset,size):
    """
        samples a dataset (a list) from a list of indices, 'picks',
        which are generated randomly as in the method get_picks.
    """
    picks = set([randint(0, size-1) for k in range(len(dataset))])
    return np.array([dataset[pick] for pick in picks])

def read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json', w=60, h=80):
    """
        reads the dataset (train and test), returns the images and labels (class and colors) for both sets
    """
    np.random.seed(123)
    ground_truth = json.load(open(gt_json, 'r'))

    train_img_names, train_class_labels, train_color_labels = [], [], []
    test_img_names, test_class_labels, test_color_labels = [], [], []
    ## IDEA: AFEGIR UNA VARIABLE TOPALL AL NÚMERO DE IMG TRAIN I test
    #        Només a debug, no a release. Per anar més ràpid. No es pot [:x]
    l1=len(ground_truth['train'].items())
    c=0
    for k, v in ground_truth['train'].items():
        printProgressBar(c + 1, l1, prefix = 'Preparing Train Data:', suffix = 'Complete', length = 50)
        c+=1
        train_img_names.append(os.path.join(ROOT_FOLDER, 'train', k))
        train_class_labels.append(v[0])
        train_color_labels.append(v[1])

    l1=len(ground_truth['test'].items())
    c=0
    for k, v in ground_truth['test'].items():
        printProgressBar(c + 1, l1, prefix = 'Preparing Test Data:', suffix = 'Complete', length = 50)
        c+=1
        test_img_names.append(os.path.join(ROOT_FOLDER, 'test', k))
        test_class_labels.append(v[0])
        test_color_labels.append(v[1])

    train_imgs, test_imgs = load_imgs(train_img_names, test_img_names)

    np.random.seed(42)

    idxs = np.arange(train_imgs.shape[0])
    np.random.shuffle(idxs)
    train_imgs = train_imgs[idxs]
    train_class_labels = np.array(train_class_labels)[idxs]
    train_color_labels = np.array(train_color_labels,dtype=object)[idxs]

    idxs = np.arange(test_imgs.shape[0])
    np.random.shuffle(idxs)
    test_imgs = test_imgs[idxs]
    test_class_labels = np.array(test_class_labels)[idxs]
    test_color_labels = np.array(test_color_labels,dtype=object)[idxs]

    return train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels


def load_imgs(train_img_names, test_img_names, w=60, h=80):
    train_imgs, test_imgs = [], []

    l1=(len(train_img_names))
    c=0
    for tr in train_img_names:
        printProgressBar(c + 1, l1, prefix = 'Reading Train Data:', suffix = 'Complete', length = 50)
        c+=1
        train_imgs.append(read_one_img(tr + '.jpg'))

    l1=(len(test_img_names))
    c=0
    for te in test_img_names:
        printProgressBar(c + 1, l1, prefix = 'Reading Test Data:', suffix = 'Complete', length = 50)
        c+=1
        test_imgs.append(read_one_img(te + '.jpg'))

    return np.array(train_imgs), np.array(test_imgs)


def read_one_img(img_name, w=60, h=80):
    img = Image.open(img_name)

    img = img.convert("RGB")

    if img.size != (w, h):
        img = img.resize((w, h))
    return np.array(img)


def visualize_retrieval(imgs, topN, info=None, ok=None, title='', query=None):
    def add_border(color):
        return np.stack([np.pad(imgs[i, :, :, c], 3, mode='constant', constant_values=color[c]) for c in range(3)],
                        axis=2)

    columns = 4
    rows = math.ceil(topN / columns)
    if query is not None:
        fig = plt.figure(figsize=(10, 8 * 6 / 8))
        columns += 1
        fig.add_subplot(rows, columns, 1 + columns)
        plt.imshow(query)
        plt.axis('off')
        plt.title(f'query', fontsize=8)
    else:
        fig = plt.figure(figsize=(8, 8 * 6 / 8))

    for i in range(min(topN, len(imgs))):
        sp = i + 1
        if query is not None:
            sp = (sp - 1) // (columns - 1) + 1 + sp
        fig.add_subplot(rows, columns, sp)
        if ok is not None:
            im = add_border([0, 255, 0] if ok[i] else [255, 0, 0])
        else:
            im = imgs[i]
        plt.imshow(im)
        plt.axis('off')
        if info is not None:
            plt.title(f'{info[i]}', fontsize=8)
    plt.gcf().suptitle(title)
    plt.show()


# Visualize k-mean with 3D plot
def Plot3DCloud(km, rows=1, cols=1, spl_id=1):
    ax = plt.gcf().add_subplot(rows, cols, spl_id, projection='3d')

    for k in range(km.K):
        Xl = km.X[km.labels == k, :]

        ax.scatter(km.centroids[:,0] , km.centroids[:,1] ,km.centroids[:,2], s = 100, color = 'k',marker=(5, 1))
        ax.scatter(Xl[:, 0], Xl[:, 1], Xl[:, 2], marker='.',
                   c=km.centroids[np.ones((Xl.shape[0]), dtype='int') * k, :] / 255)

    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    ax.set_zlabel('dim 3')
    plt.show(block=True)
    return ax


def visualize_k_means(kmeans, img_shape):
    def prepare_img(x, img_shape):
        x = np.clip(x.astype('uint8'), 0, 255)
        x = x.reshape(img_shape)
        return x

    fig = plt.figure(figsize=(8, 8))

    X_compressed = kmeans.centroids[kmeans.labels.astype('int32')]
    X_compressed = prepare_img(X_compressed, img_shape)

    org_img = prepare_img(kmeans.X, img_shape)

    fig.add_subplot(131)
    plt.imshow(org_img)
    plt.title('original')
    plt.axis('off')

    fig.add_subplot(132)

    plt.imshow(X_compressed)
    plt.axis('off')
    plt.title('kmeans')

    Plot3DCloud(kmeans, 1, 3, 3)
    plt.title('núvol de punts')
    plt.show()
