# coding: utf-8

from __future__ import print_function
import numpy as np 
np.random.seed(42)


""" 
Dummy Layer

"""
class Layer:
    
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input) # chain rule

class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output,t):
        relu_grad = input > 0
        return grad_output*relu_grad

    def backward_adam(self, input, grad_output,t):
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        

        """ 
        Define input units, output units, Learning rate and initialize weights and biases (Part A) 

        """
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale = np.sqrt(2/(input_units+output_units)), size = (input_units,output_units))
        self.biases = np.zeros(output_units)

        """
        ADAM Optimizer parameters

        """
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.m_grad = 0
        self.v_grad = 0
        self.m_bias = 0
        self.v_bias = 0
        
    def forward(self,input):
        
        return np.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output,t):

        grad_input = np.dot(grad_output, self.weights.T)
        
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input


    """ backward_adam implements ADAM optimizer (Part F) """

    def backward_adam(self,input,grad_output,t):

        grad_input = np.dot(grad_output, self.weights.T)
        
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        self.m_grad = self.beta1*self.m_grad + (1 - self.beta1)*grad_weights
        self.v_grad = self.beta2*self.v_grad + (1 - self.beta2)*(grad_weights**2)
        m_hat_grad = self.m_grad/(1 - self.beta1**t)
        v_hat_grad = self.v_grad/(1 - self.beta2**t)

        self.m_bias = self.beta1*self.m_bias + (1 - self.beta1)*grad_biases
        self.v_bias = self.beta2*self.v_bias + (1 - self.beta2)*(grad_biases**2)
        m_hat_bias = self.m_bias/(1 - self.beta1**t)
        v_hat_bias = self.v_bias/(1 - self.beta2**t)
        
        self.weights = self.weights - self.learning_rate*(m_hat_grad/(np.sqrt(v_hat_grad) + self.epsilon))
        self.biases = self.biases - self.learning_rate*(m_hat_bias/(np.sqrt(v_hat_bias) + self.epsilon))

        return grad_input
 
""" Define cost function (Part A) """        
def softmax_crossentropy(logits,reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    loss = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return loss
    
def grad_softmax_crossentropy(logits,reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    #softmax = np.exp(logits - np.max(logits)) / np.exp(logits - np.max(logits)).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]