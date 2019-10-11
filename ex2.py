from random import *
import numpy as np
import matplotlib.pyplot as plt

inpt = [[0,1],[1,0],[0,0],[1,1]]
_and = [0,0,0,1]
_or = [1,1,0,1]
nor = [0,0,1,0]
oper = [_and] + [_or] + [nor]

##Initialization

bias = -1
learning_rate = 0.1
weights = []
for i in range(3):
    weights.append(round(random(),1))


##useful functions

#function which modificates the weights during the training
def back_propagation(weights, inpl,e,op):
    inp = inpl[e]
    for j in range (len(inp)):
        errora = error(weights,inp,e,op)
        weights[j] = weights[j] + learning_rate*inp[j]*errora
        return weights
#function which gives the output of the single layer neural network
def out(weights, inp):
    out = 0
    for r in range (len(inp)):
        out = out + weights[r]*inp[r]
    out = out + weights[len(inp)]*bias
    if out>=0:
        return 1
    else:
        return 0

#function which calculates the error
def error(weights,inp,indice,op):
    true_result = oper[op][indice]
    found_result = out(weights,inp)
    error = true_result - found_result
    return error

## training

def training(size_training_examples, number_training, op, weights):
    training_examples = []
    for j in range(size_training_examples):
        training_examples = training_examples + [inpt[j]]
    for i in range(number_training):
        for j in range(size_training_examples):
            weights = back_propagation(weights, training_examples,j,op)
    return weights


## results

def result_number_training(size_training_examples,test_set, op):
    weights = []
    for i in range(3):
        weights.append(round(random(),1))
    proportion_correct_test_set =[]
    for i in range (40):
        weights = training(size_training_examples,i+1,0,weights)
        proportion =[0,0]
        for i_input in range (len(test_set)):
            if error(weights,test_set[i_input],i_input,op)!=0:
                proportion[1] = proportion[1]+1
            else :
                proportion[1] = proportion[1]+1
                proportion[0] = proportion[0]+1
        pro= proportion[0]/proportion[1]
        proportion_correct_test_set = proportion_correct_test_set + [pro]
    x = range(1,41)
    plt.plot(x,proportion_correct_test_set)
    plt.title("proportion of correct answers on number of trainings")
    plt.ylabel("proportion of correct answers")
    plt.xlabel("number of trainings")
    plt.show()
