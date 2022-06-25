import numpy as np
import torch
from scipy.special import factorial2

def relu(x):
    if x > 0:
        return x
    else:
        return 0

np_relu = np.vectorize(relu, otypes=[float])

def np_h0(x):    return np.ones(x.shape)

def np_h1(x):    return x

def np_h2(x):    return (np.power(x, 2) - 1)/np.sqrt(np.math.factorial(2))

def np_h3(x):    return (np.power(x, 3) - 3*x)/np.sqrt(np.math.factorial(3))

def np_h4(x):    return (np.power(x, 4) - 6*np.power(x, 2) + 3)/np.sqrt(np.math.factorial(4))

def np_h5(x):    return (np.power(x, 5) - 10*np.power(x, 3) + 15*x)/np.sqrt(np.math.factorial(5))

def np_h6(x): return (np.power(x, 6) - 15*np.power(x, 4) + 45*np.power(x,2) - 15)/np.sqrt(np.math.factorial(6))

def np_h7(x): return (np.power(x,7) - 21*np.power(x,5) + 105*np.power(x,3) - 105*x)/np.sqrt(np.math.factorial(7))

def np_h8(x): return (np.power(x,8) - 28*np.power(x,6) + 210*np.power(x,4) - 420*np.power(x,2) + 105)/np.sqrt(np.math.factorial(8))

def np_h9(x): return (np.power(x,9) - 36*np.power(x,7) + 378*np.power(x,5) - 1260*np.power(x,3) + 945*x)/np.sqrt(np.math.factorial(9))

def np_h10(x): return (np.power(x,10) - 45*np.power(x,8) + 630*np.power(x,6) - 3150*np.power(x,4) + 4725*np.power(x,2) - 945)/np.sqrt(np.math.factorial(10))

NP_HERMITE_POLYS = [np_h0, np_h1, np_h2, np_h3, np_h4, np_h5, np_h6, np_h7, np_h8, np_h9, np_h10]

def n_h_poly(x, order, params):
    eval = np.zeros(x.shape)
    for i in range(order+1):
        eval += params[i]*NP_HERMITE_POLYS[i](x)
    return eval

def np_hermite_poly(order):

    def fit_func(x, *params):
        eval = np.zeros(x.shape)
        for i in range(order+1):
            eval += params[i]*NP_HERMITE_POLYS[i](x)
        return eval
    
    return fit_func


def h_poly_init(order):
    k = []
    for n in range(order+1):
        if n == 0:
            k.append(1.0/np.sqrt(2*np.pi))
            #k.append(0.0)
        elif n == 1:
            k.append(1.0/2)
            #k.append(0.0)
        elif n == 2:
            k.append(1.0/np.sqrt(4*np.pi))
            #k.append(0.0)
        elif n > 2 and n % 2 == 0:
            c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
            k.append(c)
            #k.append(0.0)
        elif n >= 2 and n % 2 != 0:
            k.append(0.0)
    return k



#--------------------------------- torch activation -----------------------------------#

def t_h0(x):    return torch.ones(x.shape).to(x.device)

def t_h1(x):    return x

def t_h2(x):    return (torch.pow(x, 2) - 1)/np.sqrt(np.math.factorial(2))

def t_h3(x):    return (torch.pow(x, 3) - 3*x)/np.sqrt(np.math.factorial(3))

def t_h4(x):    return (torch.pow(x, 4) - 6*torch.pow(x, 2) + 3)/np.sqrt(np.math.factorial(4))

def t_h5(x):    return (torch.pow(x, 5) - 10*torch.pow(x, 3) + 15*x)/np.sqrt(np.math.factorial(5))

def t_h6(x): return (torch.pow(x, 6) - 15*torch.pow(x, 4) + 45*torch.pow(x,2) - 15)/np.sqrt(np.math.factorial(6))

def t_h7(x): return (torch.pow(x,7) - 21*torch.pow(x,5) + 105*torch.pow(x,3) - 105*x)/np.sqrt(np.math.factorial(7))

def t_h8(x): return (torch.pow(x,8) - 28*torch.pow(x,6) + 210*torch.pow(x,4) - 420*torch.pow(x,2) + 105)/np.sqrt(np.math.factorial(8))

def t_h9(x): return (torch.pow(x,9) - 36*torch.pow(x,7) + 378*torch.pow(x,5) - 1260*torch.pow(x,3) + 945*x)/np.sqrt(np.math.factorial(9))

def t_h10(x): return (torch.pow(x,10) - 45*torch.pow(x,8) + 630*torch.pow(x,6) - 3150*torch.pow(x,4) + 4725*torch.pow(x,2) - 945)/np.sqrt(np.math.factorial(10))

TORCH_HERMITE_POLYS = [t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_h8, t_h9, t_h10]

def t_hermite_poly(x, order, params):
    eval = torch.zeros(x.shape).to(x.device)
    for i in range(order+1):
        eval += params[i]*TORCH_HERMITE_POLYS[i](x)
    return eval

def t_hermite_poly_perNeuron(x, order, params):
    input_shape = x.shape
    res = x.view(input_shape[0], -1)
    for i in range(len(params)):
        res[:,i] = t_hermite_poly(res[:,i], order, params[i])
    return res.view(input_shape)

class hermite_poly_act(torch.nn.Module):
    def __init__(self, order, params, is_global=False):
        super().__init__()
        self.order = order
        self.params = params
        self.is_global = is_global

    def forward(self, x):
        if self.is_global:
            return t_hermite_poly(x, self.order, self.params)
        else:
            input_shape = x.shape
            res = x.view(input_shape[0], -1)
            for i in range(len(self.params)):
                res[:,i] = t_hermite_poly(res[:,i], self.order, self.params[i])
            return res.view(input_shape)