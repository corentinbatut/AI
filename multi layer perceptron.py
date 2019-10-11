from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

from datapreprocessing import X_train, y_train, X_test, y_test

laccuracy = []

for i in range(5,35,5):
    clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(X_train,y_train)
    pref= clf.predict(X_test)
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