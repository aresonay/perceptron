import numpy as np

class Neuron:
  def __init__(self, weights, bias, func):
    self.weights = weights 
    self.bias = bias 
    self.func = func
    self.input_data = [] 

  def change_bias(self, bias):
    self.bias = bias

  def choose_function(self, sigma):
    functions = {
        'relu' : self.relu(self, sigma),
        'tanh' : self.tanh(self,sigma),
        'sigmoid' : self.sigmoid(self, sigma),
        'linear': self.linear(self, sigma),
        'binarystep': self.binary_step(self, sigma)
    }
    return functions[self.func]

  def run(self, input_data:list):
    self.input_data = input_data
    x = np.array(self.input_data) 
    weights = np.array(self.weights)
    sigma = np.dot(x, weights) + (self.bias)
    return self.choose_function(sigma)

  @staticmethod
  def sigmoid(self, x):
    return 1/(1+np.exp(-x))
  @staticmethod
  def relu(self,x):
    return np.maximum(0, x)
  @staticmethod
  def tanh(self,x):
    return np.tanh(x)
  @staticmethod
  def binary_step(self,x):
    return np.heaviside(x,1)
  @staticmethod
  def linear(self, x):
    return x
        

  @property
  def show_data(self): 
    print("Weights: ", self.weights)
    print("Bias: ", self.bias)
    print("Func: ", self.func)
