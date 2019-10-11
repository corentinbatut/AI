from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

from datapreprocessing import X_train, y_train, X_test, y_test

#evaluate the performance for each number of nodes

laccuracy = []

for i in range(5,35,5):
    clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(X_train,y_train)
    pred= clf.predict(X_test)
    totgood= 0
    total = len(y_test)
    for i in range(total):
        if pred[i] == y_test[i]:
            totgood = totgood +1
    accuracy = totgood/total
    laccuracy = laccuracy +[accuracy]

plt.plot([5,10,15,20,25,30],laccuracy)
plt.title("proportion of correct answers on number of nodes (one hidden layer)")
plt.ylabel("proportion of correct answers")
plt.xlabel('number of nodes')
plt.show()

#evaluate the performance for each number of layers

laccuracy=[]
for i in range(1,11):
    l=[10]*i
    tu = tuple(l)
    clf = MLPClassifier(hidden_layer_sizes=tu)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    totgood=0
    total = len(y_test)
    for i in range(total):
        if pred[i]==y_test[i]:
            totgood=totgood+1
    accuracy=totgood/total
    laccuracy=laccuracy+[accuracy]
plt.plot(list(range(1,11)),laccuracy)
plt.title("proportion of correct answers on number of layers(10 nodes per layer)")
plt.ylabel("proportion of correct answers")
plt.xlabel("number of layers")
plt.show()

#evaluate the performance for different activation functions

laccuracy=[]
for i in ['identity','logistic','tanh','relu']:
    l=[10]*2
    tu = tuple(l)
    clf = MLPClassifier(hidden_layer_sizes=tu,activation=i)
    clf.fit(X_train,y_train)
    pred =clf.predict(X_test)
    totgood=0
    total=len(y_test)
    for i in range(total):
        if pred[i]==y_test[i]:
            totgood=totgood+1
    accuracy=totgood/total
    laccuracy=laccuracy+[accuracy]
plt.plot(['identity','logistic','tanh','relu'],laccuracy)
plt.title("proportion of correct answers on different activations functions")
plt.ylabel("proportion of correct answers")
plt.show()
