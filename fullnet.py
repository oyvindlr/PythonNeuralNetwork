import numpy as np

class activationfunction:
    """
    Class representing activation functions used in neural networks.
    
    The derivative of the activation function should also be given
    for learning.
    """
    
    def __init__(self, func, derivative) -> None:
        self.f = func
        self.df = derivative

class lossfunction:
    """
    A loss function takes two arguments; the true value/label y
    and the estimated value f(x)=yhat. The dimensions of y and yhat
    are determined by the network's output layer.
    The gradient of the cost function with relation to yhat must also be given. 
    This function also takes the values y and yhat as arguments.
    """

    def __init__(self, func, gradient):
        self.f = func
        self.df = gradient


#Functions used in activations and cost functions
def _logistic(x):
    return 1.0/(1.0+np.exp(-x))
def _logisticderivative(x):
    y = _logistic(x)
    return y*(1.0-y)
def _softmax(x):
    e = np.exp(x)
    return e/np.sum(e)
def _softmaxjacobian(x):
    o = _softmax(x)
    return np.diag(o) - np.outer(o, o)
    

######################
#Common activation functions
######################

RELU = activationfunction(lambda x: np.maximum(x, 0), lambda x: np.heaviside(x, 0.0))
"""Rectifying linear unit used for activation"""

SIGMOID = activationfunction(_logistic, _logisticderivative)
"""Sigmoid function used for activation"""

LINEAR = activationfunction(lambda x:x, lambda x: np.ones(len(x)))
"""Linear activation function - sometimes used for the final layer in regression"""

SOFTMAX = activationfunction(_softmax, _softmaxjacobian)
"""Softmax activation function used for the final layer in classification"""


######################
#Common loss functions
######################

MSE = lossfunction(lambda y, yhat:(1/len(y))*np.sum((y-yhat)**2), lambda y, yhat : (2/len(y))*(yhat - y))
"""Mean square error loss function used in backpropagation for regression"""

CROSS_ENTROPY = lossfunction(lambda y, yhat: - np.sum(y*np.log(yhat)), lambda y, yhat: -y/yhat)
"""Cross entropy loss function used in backpropagation for classification"""


class fullnet:
    """
    Class representing a neural network with methods for forward calculation
    and training/backpropagation
    """

    def __init__(self, numnodes, hiddenactivation: activationfunction = RELU, outputactivation: activationfunction = LINEAR):
        """
        numnodes: List of L integers, where L is the number of layers in the network.
                  The elements represent the number of neurons in each layer.
        hiddenactivation: Activation function used in the hidden layers, i.e.
                          all layers except the output layer. Should be an instance of
                          the class "activationfunction"
        outputactivation: Activation function used in the final layer, the output layer.
                          Should be an instance of the class "activationfunction"
        """
        self.numnodes = numnodes
        self.weights = list()
        self.hiddenactivation = hiddenactivation
        self.outputactivation = outputactivation
        self.L = len(numnodes)
        self.biases = list()
        for i in range(self.L - 1):
            self.weights.append(np.zeros((numnodes[i + 1], numnodes[i])))
            self.biases.append(np.zeros(numnodes[i + 1]))
                    
    def compute_and_return_intermediate(self, input):
        """
        Run the network forward and return activations for all layers
        """
        a = list()
        z = list()
        a.append(input)
        zc = self.weights[0] @ input + self.biases[0]        
        z.append(zc)
        for i in range(1, len(self.weights)):
            ac  = self.hiddenactivation.f(zc)
            a.append(ac)
            zc = self.weights[i] @ ac + self.biases[i]            
            z.append(zc)
        ac = self.outputactivation.f(zc)
        a.append(ac)
        return a, z
    
    def compute(self, input):
        """
        Compute the output of the network
        """
        a, dummy = self.compute_and_return_intermediate(input)
        return a[-1]
            
   
    def set_weight(self, weight, j, k, l):
        """
        Set the weight of the connection between the 
        k'th neuron of the l-1th layer and the jth neuron
        of the lth layer        
        """
        self.weights[l-1][j, k] = weight

    def set_bias(self, bias, j, l):
        """
        Set the bias of the jth neuron at the lth level
        """
        self.biases[l-1][j] = bias


    def backprop(self, x, y, loss: lossfunction):
        """
        Backpropagation finds the gradient of the cost function wrt all weights and biases in the network given a single input/output pair.
        Parameters:
            x: Input data
            y: Label
            loss: loss function (an instance of the class 'lossfunction')
        returns: a list of matrices which represent the partial derivatives wrt the weights
        and a list of vectors which represent the partial derivatives wrt the biases
        """
        a, z = self.compute_and_return_intermediate(x)
        weightsd = list() #partial derivatives wrt the weight matrices
        biasd = list() #partial derivatives wrt the biases
        
        #Compute derivatives of output layer
        activation_jacobian = self.outputactivation.df(z[-1])
        if activation_jacobian.ndim == 2:
            delta =  activation_jacobian @ loss.df(y, a[-1])
        else:
            delta =  activation_jacobian * loss.df(y, a[-1])
        w = np.outer(delta, a[-2])
        biasd.append(delta)
        weightsd.append(w)
        
        #Compute derivatives between hidden layers
        for i in range(self.L - 3, -1, -1):                            
            delta = (self.weights[i + 1].T @ delta) * self.hiddenactivation.df(z[i])
            w = np.outer(delta, a[i])
            biasd.insert(0, delta)
            weightsd.insert(0, w)
        return weightsd, biasd
    
    def cost(self, x, y, loss: lossfunction):
        c = 0.0
        for xx, yy in zip(x, y):
            t = self.compute(xx)
            c = c + loss.f(yy, t)
        return c/len(x)


    def average_gradients(self, x, y, loss):
        """
        Use backpropagation to compute the average of gradients of multiple input/output pairs.
        Parameters:
            x: list of input vectors/scalars
            y: list of output vectors/scalars
            loss: cost function
        output:
            w: list of weight matrix derivatives
            b: list of bias vector derivatives
        """
        w = list()
        b = list()
        K = self.L - 1
        for i in range(K):  
             
            w.append(np.zeros(self.weights[i].shape))
            b.append(np.zeros(self.biases[i].shape))
        for input, output in zip(x, y):
            wi, bi = self.backprop(input, output, loss)
            for wij, bij, j in zip(wi, bi, range(K)):
                w[j] = w[j] + wij
                b[j] = b[j] + bij
        n = len(x)
        for i in range(K):
            w[i] = w[i]/n
            b[i] = b[i]/n
        return w, b

    def gradient_descent_step(self, x, y, loss, stepsize):
        wgrad, bgrad = self.average_gradients(x, y, loss)
        for i in range(self.L - 1):
            self.weights[i] = self.weights[i] - stepsize*wgrad[i]
            self.biases[i] = self.biases[i] - stepsize*bgrad[i]


    def train(self, input, output, loss, stepsize, batchsize = 10, mincostchange = 1e-5, maxiter = 100):
        i = 0
        n = len(input)
        prevloss = np.inf
        curloss = self.cost(input, output, loss)
        print("Cost before training: " + str(curloss))
        while i < maxiter and np.abs(prevloss - curloss) > mincostchange:            
            curbatch = np.random.choice(n, batchsize)
            self.gradient_descent_step([input[j] for j in curbatch], [output[j] for j in curbatch], loss, stepsize)
            prevloss = curloss
            curloss = self.cost(input, output, loss)            
            print("Cost at iteration " + str(i) + ": " + str(curloss))            
            i = i + 1



        

        
            



