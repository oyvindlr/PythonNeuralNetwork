import pytest
from fullnet import *
import numpy as np
import copy

from fullnet import fullnet

@pytest.fixture
def nnetsize():
    return [4, 2, 2]

@pytest.fixture
def nnet(nnetsize: list[int]):
    return fullnet(nnetsize)

@pytest.fixture
def nnetfilled(nnetsize: list[int]):
    net = fullnet(nnetsize)
    fill_weights(net)
    return net

def test_matrices_created(nnet: fullnet, nnetsize: list[int]):
    assert len(nnet.weights) == len(nnetsize) - 1


def test_default_activation_is_relu(nnet:fullnet):
    assert nnet.hiddenactivation.f(4.0) == 4.0
    assert nnet.hiddenactivation.f(-4.0) == 0.0

def test_can_compute_output(nnet:fullnet):
    qk =  nnet.compute(np.array([1, 2, 3, 4])) 
    assert len(qk) == 2
    assert qk[0] == 0.0
    assert qk[1] == 0.0

def test_set_weight(nnet:fullnet):
    nnet.set_weight(2.5, 1, 3, 1)
    assert nnet.weights[0][1, 3] == 2.5

def test_set_bias(nnet:fullnet):
    nnet.set_bias(3.1, 1, 2)
    assert nnet.biases[1][1] == 3.1

def test_simplest_network():
    #test the simplest possible network, single input, single output, no hidden layers
    nnet = fullnet([1, 1])
    nnet.set_weight(10.3, 0, 0, 1)
    nnet.set_bias(1.2, 0, 1)
    y = nnet.compute(np.array([5.1]))
    assert y == 5.1*10.3+1.2

def test_follow_weights_and_biases_through_network(nnet:fullnet):
    nnet.set_weight(5.0, 1, 2, 1)
    nnet.set_bias(2.0, 1, 1)
    nnet.set_weight(3.0, 0, 1, 2)
    nnet.set_bias(0.5, 0, 2)
    y = nnet.compute(np.array([2.0, 3.0, 4.0, 5.0]))
    #only third value (4.0) is used
    assert y[1] == 0
    assert y[0] == (5.0*4.0+2.0)*3.0+0.5

def test_activation_in_intermediate_steps():
    #Testing that ReLU is used in the intermediate step
    nnet = fullnet([1, 2, 1], RELU, LINEAR)
    nnet.set_weight(3.0, 0, 0, 1)
    nnet.set_weight(-2.0, 1, 0, 1)
    nnet.set_weight(1.0, 0, 0, 2)
    #negative value is eradicated due to RELU, only 2.0*3.0 left
    y = nnet.compute(np.array([2.0]))
    print(y)
    assert y == 2.0*3.0

def test_activation_in_final_step():
    negate = activationfunction(lambda x:-x, None)
    nnet = fullnet([1, 1, 1, 1], LINEAR, negate)
    for i in range(1, 4):
        nnet.set_weight(1.0, 0, 0, i)
    y = nnet.compute([2.0])
    assert y == -2.0

#Test cost function
def test_cost_function():
    f = lambda y, fy: (y-fy)**2
    df = lambda y, fy: 2*fy-2*y
    cost = lossfunction(f, df)
    assert cost.f == f
    assert cost.df == df

def test_mse():
    mse = MSE.f(np.array([1.2, 2.2]), np.array([1.1, 2.0]))
    assert mse == pytest.approx(1/2*(0.1**2 + 0.2**2))

def test_mse_gradient():
    mse = MSE.df(np.array([1.2, 2.2]), np.array([1.1, 2.0]))
    assert mse[0] == pytest.approx(-0.1)
    assert mse[1] == pytest.approx(-0.2)


#Backpropagation tests

#Test the simplest possible derivative
def test_backprop_costfunction_derivative():
    nnet = fullnet([1, 1], LINEAR, LINEAR)
    nnet.set_weight(1.0, 0, 0, 1)
    [dweights, dbiases] = nnet.backprop(np.array([2.0]),np.array([1.0]), MSE)
    assert len(dweights) == 1
    assert dweights[0] == 4.0

def test_backprop_costfunction_derivative_2():
    nnet = fullnet([1, 1], LINEAR, LINEAR)
    nnet.set_weight(1.0, 0, 0, 1)
    [dweights, dbiases] = nnet.backprop(np.array([3.0]),np.array([-1.0]), MSE)
    assert dweights[0] == 24.0

def finite_differences(func, x: np.array):
    """Compute the approximate derivative/gradient/jacobian of func at x using finite differences"""
    y = func(x)
    y = np.atleast_1d(y)
    dx = 1e-5
    d = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        xdx = x.copy()
        xdx[i] = xdx[i] + dx
        dy = func(xdx)-y
        d[i,] = dy/dx    
    return np.squeeze(d)

#Test derivatives of activation functions
def test_sigmoid_derivative():
    x = np.array([0.23])
    true_deriv = SIGMOID.df(x)
    approx_deriv = finite_differences(SIGMOID.f, x)
    assert true_deriv[0] == pytest.approx(approx_deriv, 1e-5)

def test_softmax_jacobian():
    xx = np.array([0.7, 0.3, 0.1])
    true_jacobian = SOFTMAX.df(xx)
    approx_jacobian = finite_differences(SOFTMAX.f, xx)
    assert true_jacobian == pytest.approx(approx_jacobian, 1e-5)

def test_relu_derivative():
    x = np.array([0.5])
    assert RELU.df(x) == 1.0
    assert RELU.df(-x) == 0.0

def test_linear_derivative():
    x = np.array([0.5])
    assert LINEAR.df(x) == 1.0
    assert LINEAR.df(-x) == 1.0

#Test derivatives of cost functions
def test_mse_derivative():
    y = np.array([0.3, -4.2])
    yhat = np.array([-0.2, -3.3])
    approx_diff = finite_differences(lambda x: MSE.f(y, x), yhat)
    true_diff = MSE.df(y, yhat)
    assert true_diff == pytest.approx(approx_diff, 1e-4)


def test_cross_entropy_derivative():
    y = np.array([0.0, 1.0, 0.0])
    yhat = np.array([0.2, 0.6, 0.3])
    approx_diff = finite_differences(lambda x: CROSS_ENTROPY.f(y, x), yhat)
    true_diff = CROSS_ENTROPY.df(y, yhat)
    assert true_diff == pytest.approx(approx_diff, 1e-5)

def run_network_with_single_param(nnet: fullnet, vector, input):
    """Method in which all weights and biases are gathered into an array, so we can compute the gradient"""
    ind = 0
    nnetlocal = copy.deepcopy(nnet)
    for i in range(len(nnet.weights)):
        nel = nnet.weights[i].size
        nnetlocal.weights[i] = np.reshape(vector[ind:ind+nel], nnet.weights[i].shape)
        ind = ind + nel
        nnetlocal.biases[i] = vector[ind:ind + len(nnet.biases[i])]
        ind = ind + len(nnet.biases[i])
    return nnetlocal.compute(input)

def single_param_from_weights_and_biases(wa, ba):
    length = 0
    for i in range(len(wa)):
        length = length + wa[i].size + ba[i].size
    vec = np.zeros(length)
    ind = 0
    for m, b in zip(wa, ba):
        sz = m.size
        vec[ind:ind + sz] = m.reshape(sz)
        ind  = ind + sz
        vec[ind:ind + len(b)] = b
        ind = ind + len(b)
    return vec    

def single_param_from_network(nnet:fullnet):
    return single_param_from_weights_and_biases(nnet.weights, nnet.biases)

def test_single_param(nnet:fullnet):
    #test of the helper functions in this file, does not test functionality of fullnet
    fill_weights(nnet)
    input = np.array([1.0, 2.0, 3.0, 4.0])
    y = nnet.compute(input)
    v = single_param_from_network(nnet)
    y2 = run_network_with_single_param(nnet, v, input)
    assert list(y) == list(y2)



def backprop_test_general(dims, hidden_activation, output_activation, cost, input, output):
    nnet = fullnet(dims, hidden_activation, output_activation)
    rng = fill_weights(nnet)
    w, b = nnet.backprop(input, output, cost)
    weightsvec = single_param_from_network(nnet)
    func_of_weights = lambda x: cost.f(output, run_network_with_single_param(nnet, x, input))
    grad_approx = finite_differences(func_of_weights, weightsvec)
    grad_exact = single_param_from_weights_and_biases(w, b)
    return grad_exact, grad_approx

def test_backpropagation_simple():
    """Compare the gradient calculated by backpropagation to the one calculated by finite differences"""    
    input = np.array([-2.0])
    output = np.array([3.1])
    grad_exact, grad_approx = backprop_test_general([1, 1], None, SIGMOID, MSE, input, output)
    assert grad_exact == pytest.approx(grad_approx, 1e-4)
    

def test_backpropagation_mimo_no_hidden_layers():
    input = np.array([-2.0, 1.5])
    output = np.array([3.1, -0.5])
    grad_exact, grad_approx = backprop_test_general([2, 2], None, SIGMOID, MSE, input, output)
    assert grad_exact == pytest.approx(grad_approx, 1e-4)


def test_backpropagation_one_hidden_layer():
    """Compare the gradient calculated by backpropagation to the one calculated by finite differences"""
    input = np.array([-2.0])
    output = np.array([3.1])
    grad_exact, grad_approx = backprop_test_general([1, 1, 1], SIGMOID, LINEAR, MSE, input, output)
    assert grad_exact == pytest.approx(grad_approx, 1e-4)

def test_backpropagation_advanced_network():
    """Compare the gradient calculated by backpropagation to the one calculated by finite differences"""    
    input = np.array([-1.0, 2.0, 2.3])
    output = np.array([3.1, -2.3, 3.2])
    grad_exact, grad_approx = backprop_test_general([3, 2, 2, 3], SIGMOID, LINEAR, MSE, input, output)
    assert grad_exact == pytest.approx(grad_approx, abs= 1e-5)    


def test_backprop_with_jacobian():
    """When the output of the i'th neuron of the final activation function
    is not independent of the input to the j'th neuron for j != i, the derivative
    of the activation function must be a Jacobian matrix instead of an array.
    This happens for the softmax function. Here we test this special case."""
    input = np.array([-1.0, 2.0, 2.3])
    output = np.array([3.1, -2.3, 3.2])
    grad_exact, grad_approx = backprop_test_general([3, 2, 2, 3], SIGMOID, SOFTMAX, CROSS_ENTROPY, input, output)
    assert grad_exact == pytest.approx(grad_approx, abs= 1e-5)    

def test_average_gradient(nnetfilled:fullnet):
    input = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([-2.0, 3.0, 4.0, -1.0])]
    output = [np.array([3.2, 4.1]), np.array([-2.3, -3.0])]
    wa, ba = nnetfilled.backprop(input[0], output[0], MSE)
    wb, bb = nnetfilled.backprop(input[1], output[1], MSE)
    #Just pick some random indices to check
    avgw = (wa[1][0, 1]+wb[1][0, 1])/2
    avgb = (ba[0][1]+bb[0][1])/2
    
    w, b = nnetfilled.average_gradients(input, output, MSE)
    
    assert w[1][0, 1] == avgw
    assert b[0][1] == avgb

def test_gradient_descent(nnetfilled:fullnet):
    input = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([-2.0, 3.0, 4.0, -1.0])]
    output = [np.array([3.2, 4.1]), np.array([-2.3, -3.0])]
    w, b = nnetfilled.average_gradients(input, output, MSE)
    
    stepsize = 2.5
    expectedw = nnetfilled.weights[1][0, 1] - stepsize * w[1][0, 1]
    expectedb = nnetfilled.biases[0][1] - stepsize * b[0][1]

    nnetfilled.gradient_descent_step(input, output, MSE, stepsize)

    assert expectedb == nnetfilled.biases[0][1]
    assert expectedw == nnetfilled.weights[1][0, 1]

def test_train_regression(nnetfilled:fullnet):
    """Tests only if training cost is reduced for some random data.
      Better tests require too much code and advanced concepts for a unit test"""
    rng = np.random.default_rng(seed = 0)
    x = list()
    y = list()
    for i in range(10):
        x.append(rng.normal(scale = 10.0, loc = 1.0, size = 4))
        y.append(rng.normal(scale = 3.0, loc = -2.0, size =2))
    cost_before = nnetfilled.cost(x, y, MSE)

    nnetfilled.train(x, y, MSE, 0.00001, 2, maxiter = 30)

    cost_after = nnetfilled.cost(x, y, MSE)

    assert cost_after < cost_before



def fill_weights(nnet:fullnet):
    rng = np.random.default_rng(seed = 0)
    for i in range(len(nnet.weights)):
        nnet.weights[i] = rng.normal(scale = 10.0, size = nnet.weights[i].shape)
        nnet.biases[i] = rng.normal(scale = 2, size = len(nnet.biases[i]))


#TODO:
        #Support for sparse weights
        #Better optimization
        #Support for regularization - general or per cost func?
        #More activations and cost functions
        #More documentation for functions
        #Not planned: Normalization layers and skip connections        