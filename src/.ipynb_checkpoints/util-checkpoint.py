import numpy as np

# ====================================================
# layer; layer methods, which related parameter update
# ====================================================

def Linear(W, a, b=None):
    """
    Args:
        W : weight parameter
        a : vector
        b : bias
    return:
        dots of W and a 
    """

    if b is None:
        return np.dot(W, a)
    else:
        return np.dot(W, a) + b

def Aggregate(W, X_0, adj):
    """
    Args:
        W : Weight Parameter
        X_0 : feature matrix
        adj : adjacent matrix
    return:
        ret: aggregated feature matrix
    """
    A = [] # Sum of adjacent node's feature vectors of each nodes 
    for i in range(len(X_0)):
        tmp = np.zeros((len(X_0[i])))
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                tmp += np.array(X_0[j])
        A.append(tmp)
    ret = [ReLU(Linear(W, A[i])) for i in range(len(A))]
    return np.array(ret)

def Aggregate_t_times(W, adj, embed_dim, t_hop=2):
    # initialize feature value
    X_t = np.zeros((len(adj), embed_dim))
    X_t[:, 0] = 1.
    # aggregate t_hop times
    for _ in range(t_hop):
        ag = Aggregate(W, X_t, adj)
        X_t = ag
    return Readout(X_t)

# =================================================================
# functions; other functions, which is not related parameter update
# =================================================================

def ReLU(x):
    """
    Args:
        x : vector
    return:
        x : activated vector
    """
    return np.maximum(0, x)

def Readout(X):
    """
    Args:
        X : matrix
    return:
        x : readouted vector
    """
    return np.sum(X, axis=0)

def Sigmoid(x):
    """
    Args:
        x : vecotr
    return:
        values between 0 to 1
    """
    if x < 0:
        a = np.exp(x)
        return a / (1 + a)
    else:
        return 1 / (1 + np.exp(-x))

# =============================
# loss function; calculate loss
# =============================

def cross_entropy(y_hat, y):
    """
    Args:
        y_hat : output value
        y : target value
    return:
        binary cross entropy error
    """
    return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

def cross_entropy_enhanced(s, y, threshold=5):
    """
    to avoid overflow while caluclating loss, 
    individualy process to each conditions

    Args:
        s : Activated(Assumed as Activated with Sigmoid function) value
        y : target value 
        threshold : threshold value
    return: 
        adjusted binary cross entropy error
    """
    if s < -threshold:
        return y * -s
    elif s > threshold:
        return (1 - y) * s
    else:
        y_hat = Sigmoid(s)
        return cross_entropy(y_hat, y)

# ===========================
# optimizer; SGD and Momentum
# ===========================

# def SGD(params, grads, lr=0.01):
#     """
#     Stochastic Gradient Descent
#     """
#     for i in range(len(params)):
#         params[i] -= lr * grads[i]
#     return params

# def Momentum(params, grads, lr=0.01, momentum=0.9):
#     """
#     Momentum SGD
#     """
#     w = []
#     for param in params:
#         w.append(np.zeros_like(param))
#     for i in range(len(params)):
#         w[i] = momentum * w[i] - lr * grads[i]
#         params[i] += w[i]
#     return params

from copy import deepcopy

class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.0001):
        self.lr = lr
        
    def update(self, params, grads):
        # params = deepcopy(params)
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
        return params


class Momentum:
    """
    Momentum SGD
    """
    def __init__(self, lr=0.0001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.w = None
        
    def update(self, params, grads):
        # params = deepcopy(params)
        if self.w is None:
            self.w = []
            for param in params:
                self.w.append(np.zeros_like(param))

        for i in range(len(params)):
            self.w[i] = self.momentum * self.w[i] - self.lr * grads[i]
            params[i] += self.w[i]
        return params

class Adam:
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param, dtype='float64'))
                self.v.append(np.zeros_like(param, dtype='float64'))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)