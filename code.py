import numpy as np 
import matplotlib.pyplot as plt

def designer(x1, x2): #plot x1  sur l'axe de X et x2 sur l'axe Y
    plt.plot(x1, x2)
	
def sigmoid(score): # sigmoid function
    return 1/(1+np.exp(-score))
	
def calcule_erreur (ligne_parameters, points, y): #cross entropy erreur de classification binaire =-somme(Y*log(P) + (1-Y) log(1-P))
    N = points.shape[0] # nomble des exmaples inputed_data
    P = sigmoid(points * ligne_parameters) # propabilité de inputed_data si P> 0.5 reference à la première classe (1) sinon reference à la deuscième classe (0) 
    #print(P)
    cross_entropy =-(1/N)*(np.log(P).T * y + np.log(1-P).T*(1-y))
    return cross_entropy
	
def gradiant_descent (ligne_parameters, points, y, alpha): #gradiant descent (pour corriger les paramètres de réseau de neurones) = 1/N*(P-Y)*inputed_data * learning_rate(alpha)
    N=points.shape[0]
    for i in range(2000): # nombre d'itération pour corriger (w1, w2, b)
        P=sigmoid(points * ligne_parameters) 
        gradiant = (points.T*(P - y))*(alpha/N)
        ligne_parameters = ligne_parameters - gradiant
        #print(ligne_parameters.shape)
        w1 = ligne_parameters.item(0)
        w2 =ligne_parameters.item(1)
        b= ligne_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max() ])
        x2= -b/w2 + x1 * (-w1/w2)
    designer(x1, x2) #designer la ligne de séparation de deux classes 
        
# pour executer ce code sur jupyter copie la partie des fonctions dans une cell et les lignes qui se situe au dessous de cette ligne dans une autre cell (sinon faire rien)

n_pts = 100
np.random.seed(0) # afin de garder les memes random numbers 

#inputed_data
bias = np.ones(n_pts) 
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T #deuscième classe (0)
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T #premiere classe (1)
points= np.vstack((top_region, bottom_region)) # array de dimension (n_pts*2, 3) 

#designer une test ligne
#w1 = -0.2
#w2 = -0.35
#b=3.5
#ligne_parameters = np.matrix([w1, w2, b]).T
#x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max() ])
#x2= -b/w2 + x1 * (-w1/w2)
#print (x1, x2)
#combinaison_lineare = points * ligne_parameters
ligne_parameters = np.matrix([np.zeros(3)]).T
Y= np.array([0]*n_pts+[1]*n_pts).reshape(n_pts*2, 1) # output data (traget data) qu'on veut predicter 
#Y
#probabilite


_, ax =plt.subplots(figsize = (4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r') #deuxièe classe en rouge
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')#premier classe en blue
#designer(x1, x2)
gradiant_descent(ligne_parameters, points, Y, 0.06)
calcule_erreur(ligne_parameters, points, Y)

