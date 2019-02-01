import scipy.io
import numpy as np
import random as rd
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.data import loadlocal_mnist


'''
Définition de la classe RBM
'''
class RBM:

    def __init__(self,W, B, A):
        self.W = W
        self.B = B
        self.A = A

    @property
    def W(self):
        return self.__W

    @property
    def B(self):
        return self.__B
    @property
    def A(self):
        return self.__A
    @W.setter
    def W(self, W):
        self.__W = W
    @B.setter
    def B(self, B):
        self.__B = B
    @A.setter
    def A(self, A):
        self.__A = A

def lire_alpha_digit(file):
    X = pd.DataFrame(scipy.io.loadmat(file)['dat'])
    images = pd.DataFrame(np.zeros((1404, 20*16), dtype=np.int8))
    ligne = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            images.iloc[ligne,:] = np.concatenate([X.iloc[i,j][k] for k in range(20)])
            ligne += 1
    return images.as_matrix()

def init_rbm(p,q):
    B = np.zeros(q)
    A = np.zeros(p)
    rbm = RBM(W=0.1 * np.random.randn(p, q), B=B, A=A)
    return rbm

def sigmoid(x):
    return(1/(1+np.exp(-x)))


'''
Input: 
    X : une matrice image
    RBM : objet de class RBM, contenant W, A, B

Output:
    PH|V : une matrice de probabilités
'''

def entree_sortie_RBM(X,RBM):
    B = np.array([RBM.B]*X.shape[0])
    return sigmoid(np.dot(X,RBM.W) - B)


'''
Input:
    H : matrice de probabilités
    RBM : objet de class RBM, contenant W, A, B

Output:
    PV|H : matrice de probabilités
'''

def sortie_entree_RBM(H,RBM):
    A = np.array([RBM.A]*H.shape[0])
    return sigmoid(np.dot(H,np.transpose(RBM.W)) + A)


'''
Input: 
    X : matrice image
    RBM : objet de class RBM, contenant W, A, B
    batch_size : int, taille de nos batchs
    nb_iteration : int, nombre d'itérations
    alpha : le learning rate

Output:
    un objet RBM entrainé et une liste contenant les erreurs quadratiques moyennes
'''


def train_RBM(X, RBM, batch_size, nb_iteration, alpha):
    EQM = []
    for iteration in range(nb_iteration):
        # shuffle inplace
        l = np.arange(X.shape[0])
        np.random.shuffle(l)
        p, q = RBM.W.shape
        for j in range(0, X.shape[0], batch_size):
            # selection des données
            batch_ind = l[j:min(j + batch_size, X.shape[0])]
            data = X[batch_ind, :]
            n = len(data)

            # CD_1
            # tirage selon H|V0. Donne un tableau de False/True, mais numpy sait que c'est des 0 ou 1

            V0 = data
            PH_V0 = entree_sortie_RBM(V0, RBM)
            p, q = PH_V0.shape
            H0 = np.random.rand(p, q) <= PH_V0

            PV_H0 = sortie_entree_RBM(H0, RBM)
            p, q = PV_H0.shape
            V1 = np.random.rand(p, q) <= PV_H0
            PH_V1 = entree_sortie_RBM(V1, RBM)

            # calcul des gradients
            dW = (np.dot(np.transpose(V0), PH_V0) - np.dot(np.transpose(V1), PH_V1)) / data.shape[0]
            # somme sur les lignes
            dA = (np.sum(V0, axis=0) - np.sum(V1, axis=0)) / data.shape[0]
            dB = (np.sum(PH_V0, axis=0) - np.sum(PH_V1, axis=0)) / data.shape[0]

            # Actualisation
            RBM.W += alpha * dW
            RBM.A += alpha * dA
            RBM.B += alpha * dB

        e = entree_sortie_RBM(X, RBM)
        s = sortie_entree_RBM(e, RBM)
        erreur = np.mean(np.mean(X - s) ** 2)
        print(erreur)
        EQM.append(erreur)
    tracer_erreur(EQM)

    return RBM


'''
Input: 
    EQM : une liste contenant les erreurs quadratiques moyennes à chaque itération
Ouput: 
    Un graphe de l'evolution de l'erreur quadratique moyenne
'''
def tracer_erreur(EQM):
    plt.plot(range(len(EQM)), EQM)
    plt.legend(['Erreur Quadratique Moyenne'])
    plt.title("Évolution de l'erreur")
    plt.savefig('erreur-quadra.png')


'''
Input: 
    images : une matrice(np array) contenant une image sur chaque lignes
    size : la taille des images
Output:
    affiche les images
'''
def display(images, size):
    for image in images:
        image = image*255
        plt.imshow(np.reshape(image, size),cmap='gray')
        plt.show()


'''
Input: 
    RBM : un RBM entrainé
    nb_iterations : le nombre d'iterations
    nb_images : le nombre d'images à générer
    size_image : la taille souhaitée des images à générer
Ouput: 
    nb_images images générées
'''


def generer_image_RBM(RBM, nb_iterations, nb_images, size_image):
    images = []
    # initialisation
    images = np.zeros((nb_images, RBM.W.shape[0]))
    for i in range(nb_images):
        X = np.random.rand(nb_images, RBM.W.shape[0])
        comp = np.ones((nb_images, RBM.W.shape[0])) * 0.5
        X = X <= comp
        for iteration in range(nb_iterations):
            PH_V = entree_sortie_RBM(X, RBM)
            p, q = PH_V.shape
            H = (np.random.rand(p, q)) <= (PH_V)

            PV_H = sortie_entree_RBM(H, RBM)
            p, q = RBM.W.shape

            X = (np.random.rand(nb_images, p)) <= (PV_H)
        images = X
    return images


'''
Test des fonctions précédentes sur les données Binary AlphaDigits
Output : les images générées
'''


def principal_RBM_Alpha():
    # Chargement des images
    X = lire_alpha_digit('./binaryalphadigs.mat')

    # Initialisation des paramètres
    p = X.shape[1]
    q = 60
    batch_size = 9
    nb_iter = 1000
    nb_iter_gibbs = 1000
    nb_image = 10
    alpha = 0.1
    taille_image = [20, 16]

    #  On mélange les données et on en sélectionne seulement 100
    np.random.shuffle(X)
    X = X[:99, :]

    # On initialise le RBM
    rbm = init_rbm(p, q)

    # On l'entraine
    rbm_train, EQM = train_RBM(X, rbm, batch_size, nb_iter, alpha)

    # On affiche l'évolution de l'erreur de reconstruction
    tracer_erreur(EQM)

    # On génére nb_image images
    images = generer_image_RBM(rbm_train, nb_iter_gibbs, nb_image, taille_image)

    return (images)


'''
Input: 
    size: une liste contenant la taille des couches des RBM composants le DNN
Output:
    Un DNN
'''
def init_DNN(size):
    DNN = []
    for i in range(len(size)-1):
        DNN.append(init_rbm(size[i], size[i+1]))
    return DNN


'''
Input: 
    DNN : un DNN initialisé
    nb_iter : le nombre d'itérations
    alpha : le learning rate
    minibatch_size : la taille du minibatch
    X : une matrice d'images
Output:
    DNN_New : un DNN entrainé
'''


def train_DBN(DNN, nb_iter, alpha, minibatch_size, X):
    dnn = []
    donnees = X
    for i in range(len(DNN)):
        dnn.append(train_RBM(donnees, DNN[i], minibatch_size, nb_iter, alpha))
        donnees = entree_sortie_RBM(donnees, DNN[i])
    return dnn


'''
Input: 
    DBN : un DBN entrainé
    nb_iterations : le nombre d'itérations
    nb_images : le nombre d'images à générer
    size_image : la taille souhaitée des images
Output:
    nb_images images générées
'''


def generer_image_DBN(DBN, nb_iterations, nb_images, size_image):
    images = []
    images = np.zeros((nb_images, DBN[0].W.shape[0]))
    for i in range(nb_images):
        X = np.random.rand(nb_images, DBN[0].W.shape[0])
        comp = np.ones((nb_images, DBN[0].W.shape[0])) * 0.5
        X = X <= comp
        for iteration in range(nb_iterations):

            for RBM in DBN:
                PH_V = entree_sortie_RBM(X, RBM)
                p, q = PH_V.shape
                X = (np.random.rand(p, q)) <= (PH_V)

            for RBM in DBN[::-1]:
                PV_H = sortie_entree_RBM(X, RBM)
                p, q = RBM.W.shape
                X = (np.random.rand(nb_images, p)) <= (PV_H)
        images = X
    return images


'''
Test des fonctions précédentes sur les données Binary AlphaDigits
Output : les images générées
'''


def principal_DBN_alpha():
    # Chargement des images
    X = lire_alpha_digit('./binaryalphadigs.mat')

    # Initialisation des paramètres
    size = [320, 160, 80, 10]
    minibatch_size = 9
    nb_iter = 1000
    nb_image = 10
    alpha = 0.1
    size_image = [20, 16]

    #  On mélange les données et on en sélectionne seulement 100
    np.random.shuffle(X)
    X = X[:99, :]

    # On initialise le RBM
    DBN = init_DNN(size)

    # On l'entraine
    DBN_train = train_DBN(DBN, nb_iter, alpha, minibatch_size, X)

    # On génére nb_image images
    images = generer_image_DBN(DBN_train, nb_iter, nb_image, size_image)

    return images

'''
Input:
    data_name : fichier de données
    label_name : fichier des labels
    nb_data : nombre d'images voulues
Output:
    4 matrices: X_train, X_test, Y_train, Y_test
'''
def load_data(data_name, label_name, nb_data):
    train,train_labels = loadlocal_mnist(
    images_path=data_name,
    labels_path=label_name)
    train = train[:nb_data, :]
    train_labels = train_labels[:nb_data]
    X_train, X_test, Y_train, Y_test = train_test_split(train, train_labels, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

'''
Input: une matrice d'images en couleur
Output: une matrice d'images en noir et blanc
'''
def convert_image(X):
    comp = np.ones((X.shape[0], X.shape[1]))*127
    X = X <= comp
    return X


'''
Input:
    RBM: un RBM initialisé
    X: une matrice d'images
Output:
    Matrice de probabilités de sortie à partir de la fonction softmax
'''
def calcul_softmax(RBM,X):
    B = np.array([RBM.B]*X.shape[0])
    numerator = np.exp(np.dot(X, RBM.W) + B)
    denum = numerator.sum(1)
    result = np.zeros(numerator.shape)
    for i in range(result.shape[1]):
        result[:,i] = denum
    return numerator/result


'''
Input: 
    DNN : un DNN entrainé
    X : une matrice d'images
Output:
    sorties : sorties de toutes les couches du réseau
    probas : sortie de la dernière couche (prédiction)
'''


def entree_sortie_reseau(DNN, X):
    sorties_couches = []
    sorties_couches.append(entree_sortie_RBM(X, DNN[0]))
    for i in range(1, len(DNN)-1):
        sorties_couches.append(entree_sortie_RBM(sorties_couches[-1], DNN[i]))
    sorties_couches.append(calcul_softmax(DNN[-1], sorties_couches[-1]))
    return sorties_couches


'''
Input:
    DNN : un DNN initialisé (et?) entrainé
    nb_iter_grad : nombre d'itérations de la descente de gradient
    mini_batch_size : taille du minibatch
    X : une matrice d'images
    label : les labels de nos données
Output:
    DNN : Un DNN ayant subit une rétropropagation
    Entropie : liste des entropies croisées pour chaque itération
'''
def copy_dnn(dnn):
    network_layers = [784, 360, 100, 10]
    new_dnn = init_DNN(network_layers)
    for i in range(len(dnn)):
        new_dnn[i].A = dnn[i].A.copy()
        new_dnn[i].B = dnn[i].B.copy()
        new_dnn[i].W = dnn[i].W.copy()
    return new_dnn

def retropropagation(DNN, nb_iter_grad, alpha, mini_batch_size, X, label, test_mode=False):
    Entropie = []
    for iteration in range(nb_iter_grad):

        # Pour shuffle X, on a aussi besoin de shuffle les labels. Donc on shuffle les indices, et on récupère les données dans cet ordre
        l = np.arange(X.shape[0])
        np.random.shuffle(l)


        n = len(X)
        for batch in range(0, X.shape[0], mini_batch_size):
            new_dnn = copy_dnn(DNN)
            # selection des données
            batch_ind = l[batch:min(batch + mini_batch_size, X.shape[0])]
            donnees_batch = X[batch_ind, :]
            sorties_couches = entree_sortie_reseau(DNN, donnees_batch)

            matrice_c = sorties_couches[-1] - label[batch_ind]
            der_W = np.dot(sorties_couches[len(DNN) - 2].transpose(), matrice_c) / donnees_batch.shape[0]
            der_b = matrice_c.sum(0) / donnees_batch.shape[0]
            new_dnn[-1].W = new_dnn[len(DNN) - 1].W - alpha * der_W  # /batch
            new_dnn[-1].B = new_dnn[len(DNN) - 1].B - alpha * der_b

            for i in range(len(DNN) - 2, -1, -1):

                if i == 0:
                    data = donnees_batch
                else:
                    data = sorties_couches[i - 1]

                h_1_h = sorties_couches[i] * (1 - sorties_couches[i])
                transit = np.dot(matrice_c, np.transpose(DNN[i + 1].W))
                matrice_c = transit * h_1_h

                der_W = np.dot(np.transpose(data), matrice_c) / donnees_batch.shape[0]
                der_b = np.sum(matrice_c, axis=0) / donnees_batch.shape[0]

                new_dnn[i].W = new_dnn[i].W - alpha * der_W
                new_dnn[i].B = new_dnn[i].B - alpha * der_b

            DNN = copy_dnn(new_dnn)
        sorties_couches = entree_sortie_reseau(DNN, X)
        entropie = -np.sum(np.sum(label * np.log10(sorties_couches[-1]), axis=0), axis=0)
        print(entropie)
        Entropie.append(entropie)

    return DNN, Entropie


'''
Input : Liste des entropies croisées
Output : trace l'évolution de l'entropie
'''
def tracer_entropie(Entropie, Entropie2):
    entrepie1, = plt.plot(range(len(Entropie)), Entropie, 'b')
    entropie2, = plt.plot(range(len(Entropie2)), Entropie2, 'g')
    plt.legend([entrepie1, entropie2], ["Entropie avec pre entrainement", "Entropie sans pre entrainement"])
    plt.title("Évolution de l'Entropie")
    plt.savefig('entropie.png')


'''
Input:
    DNN : un DNN entrainé et/ou ayant subit une rétropropagation
    X_test : la matrice des données/images test
    label_test : les labels associés
Output:
    Le taux de bonne prédiction
'''
def test_DNN(DNN, X_test, label_test):
    sorties_couches = entree_sortie_reseau(DNN, X_test)
    index_max = np.argmax(sorties_couches[-1], axis=1)
    accuracy_rate = 0
    for i, j in enumerate(index_max):
        values = np.zeros((DNN[-1].W.shape[1]))
        values[j] = 1
        sorties_couches[-1][i,:] = values
    for labels in range(label_test.shape[0]):
        if (sorties_couches[-1][labels, :]==label_test[labels]).all():
            accuracy_rate += 1
    return accuracy_rate/label_test.shape[0]*100

def tranform_to_dummy(matrice):
    dummy_matrice = np.zeros((matrice.shape[0], max(matrice) + 1))
    for i in range(len(matrice)):
        dummy_matrice[i, matrice[i]] = 1
    return dummy_matrice

def principal_DNN_MNIST():
    # Initialisation des paramètres
    nb_iter_train = 100
    nb_iter = 200
    alpha = 0.1
    alpha_retro = 0.1
    mini_batch_size = 100
    nb_data_train = 1000
    image_size = [28, 28]
    nb_images = 10
    nb_data_test = 20

    X, y = loadlocal_mnist(
        images_path='./train-images.idx3-ubyte',
        labels_path='./train-labels.idx1-ubyte')

    X = np.array(X >= 127, dtype=np.int8)

    Y = tranform_to_dummy(y)

    X_test, Y_test = loadlocal_mnist(
        images_path='./t10k-images.idx3-ubyte',
        labels_path='./t10k-labels.idx1-ubyte')

    Y_test = tranform_to_dummy(Y_test)

    # Initialisation de la taille du réseau
    network_layers = [X.shape[1], 500, 500, 500, 10]

    # On initialise le RBM
    dbn = init_DNN(network_layers)
    dbn_without_pre_train = init_DNN(network_layers)
    X_train = X
    Y_train = Y

    X_test = X_test
    Y_test = Y_test

    # On l'entraine
    dnn = train_DBN(dbn, nb_iter_train, alpha, mini_batch_size, X_train)


    # On applique la rétro propagation
    dnn_result, entropie = retropropagation(dnn, nb_iter, alpha_retro, mini_batch_size, X_train, Y_train, test_mode=False)

    dnn_result_without_pre_train, entropie_w_pre_train = retropropagation(dbn_without_pre_train, nb_iter, alpha_retro, mini_batch_size, X_train, Y_train, test_mode=False)


    rate = test_DNN(dnn_result, X_test, Y_test)
    rate_without_pre_train = test_DNN(dnn_result_without_pre_train, X_test, Y_test)
    print("Accuracy pre trained: {}%".format(rate))

    print("Accuracy without pre trained: {}%".format(rate_without_pre_train))
    tracer_entropie(entropie, entropie_w_pre_train)


if __name__ == '__main__':
    principal_DNN_MNIST()