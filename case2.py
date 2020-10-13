from math import exp 

from random import seed 

from random import random 

import matplotlib.pyplot as plt 

import numpy as np 

  

# Initialize a network 

def initialize_network(n_inputs, n_hidden, n_outputs): 

    network = list() 

    hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)] 

    network.append(hidden_layer) 

    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)] 

    network.append(output_layer) 

    print(network) 

    return network 

  

# Calculate neuron activation for an input 

def activate(weights, inputs): 

    activation = weights[-1] 

    for i in range(len(weights)-1): 

        activation += weights[i] * inputs[i] 

    return activation 

  

# Transfer neuron activation 

def transfer(activation): 

    return 1.0 / (1.0 + exp(-activation)) 

  

# Forward propagate input to a network output 

def forward_propagate(network, row): 

    inputs = row 

    for layer in network: 

        new_inputs = [] 

        for neuron in layer: 

            activation = activate(neuron['weights'], inputs) 

            neuron['output'] = transfer(activation) 

            new_inputs.append(neuron['output']) 

        inputs = new_inputs 

    return inputs 

  

# Calculate the derivative of an neuron output 

def transfer_derivative(output): 

    return output * (1.0 - output) 

  

# Backpropagate error and store in neurons 

def backward_propagate_error(network, expected): 

    for i in reversed(range(len(network))): 

        layer = network[i] 

        errors = list() 

        if i != len(network)-1: 

            for j in range(len(layer)): 

                error = 0.0 

                for neuron in network[i + 1]: 

                    error += (neuron['weights'][j] * neuron['delta']) 

                errors.append(error) 

        else: 

            for j in range(len(layer)): 

                neuron = layer[j] 

                errors.append(expected[j] - neuron['output']) 

        for j in range(len(layer)): 

            neuron = layer[j] 

            neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) 

  

# Update network weights with error 

def update_weights(network, row, l_rate): 

    for i in range(len(network)): 

        inputs = row[1:] 

        if i != 0: 

            inputs = [neuron['output'] for neuron in network[i - 1]] 

        for neuron in network[i]: 

            for j in range(len(inputs)): 

                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] 

            neuron['weights'][-1] += l_rate * neuron['delta'] 

  

# Train a network for a fixed number of epochs 

def train_network(network, train, l_rate, n_epoch, n_outputs): 

    sum_errors=0 

    errors=list() 

    while sum_errors<1: 

        for epoch in range(n_epoch): 

            sum_error = 0 

            for row in train: 

                #print(row) 

                outputs = forward_propagate(network, row) 

                #print(outputs) 

                expected = [0 for i in range(n_outputs)] 

                expected[row[0]] = 1 

                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])/2 

                backward_propagate_error(network, expected) 

                update_weights(network, row, l_rate) 

            if epoch!=0: 

                sum_errors+=sum_error 

            if epoch!=0 and epoch%100==0: 

                errors.append(sum_errors*2) 

                ep.append(epoch) 

                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_errors*2)) 

                sum_errors=0 

    return errors 

         

def predict(network, row): 

    outputs = forward_propagate(network, row) 

    return outputs.index(max(outputs)) 

  

# Backpropagation Algorithm With Stochastic Gradient Descent 

def back_propagation(train, test, l_rate, n_epoch, n_hidden): 

    scores=list() 

    n_inputs = 27 

    n_outputs = len(set([row[0] for row in train])) 

    network = initialize_network(n_inputs, n_hidden, n_outputs) 

    sum_errors=train_network(network, train, l_rate, n_epoch, n_outputs) 

    predictions = list() 

    #for row in test: 

    #   prediction = predict(network, row) 

    #    predictions.append(prediction) 

    #    accuracy=accuracy_metric(row[0],predictions) 

    #    scores.append(accuracy) 

    #return(scores) 

    return sum_errors 

  

def accuracy_metric(actual, predicted): 

    correct = 0 

    for i in range(actual): 

        print(actual) 

        print(predict[i]) 

        if actual == predicted: 

            correct += 1 

    return correct / float(actual) * 100.0 

  

# Test training backprop algorithm 

#dataset=open('C:/Users/Rajitha Bhavani/Documents/upsampled.txt','r') 

#dataset=[[0,5.239742700000001,-6.3819862999999994],[1,9.716892,-9.001553900000001]]; 

#test=[[0,7.937372200000001,-7.651209599999999],[1,2.7600646,-2.6970943]]; 

ep=list() 

#sum_errors=back_propagation(dataset,test,0.1,1000,55) 

scores=list() 

n_inputs = 27 

n_outputs = len(set([row[0] for row in dataset])) 

network = initialize_network(n_inputs, 55, n_outputs) 

sum_errors=train_network(network, dataset, 0.1, 4000, n_outputs) 

TN=0 

TP=0 

FN=0 

FP=0 

for row in test: 

    prediction = predict(network, row) 

    if row[0]==0: 

        if prediction==row[0]: 

            TN+=1 

        else: 

            FN+=1 

    if row[0]==1: 

        if prediction==row[0]: 

            TP+=1 

        else: 

            FP+=1 

print(TN,FN,TP,FP) 

if TP!=0 or FN!=0: 

    SN=TP/TP+FN 

    print(SN) 

if TN!=0 or FP!=0: 

    SP=TN/TN+FP 

    print(SP) 

plt.plot(ep,sum_errors)  

plt.show() 

#print('Scores: %s' % scores) 

#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 