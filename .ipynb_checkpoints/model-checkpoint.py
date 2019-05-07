import numpy as np

class GNN(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = len(X)
        # initialize params
        self.W = np.random.normal(0, 0.4, (self.input_dim, self.input_dim))
        self.A = np.random.normal(0, 0.4, self.input_dim)
        self.b = 0
        
    def relu(self, x):
        return np.maximum(0, x)

    def aggregate(self, W, X):
        A = [np.sum(X, axis=0) - X[i] for i in range(len(X))]
        ret = [self.relu(np.dot(W, A[i])) for i in range(len(A))]
        return np.array(ret)

    def readout(self, X):
        return np.sum(X, axis=0)
    
    def forward(self, t_hop, W, X):
        """
        forward method.
        """
        X_t = X
        for t in range(t_hop):
            ag = self.aggregate(W, X_t)
            X_t = ag
        return self.readout(X_t)
    
    def sigmoid(self, x):
        if x < 0:
            a = np.exp(x)
            return a / (1 + a)
        else:
            return 1 / (1 + np.exp(-x))
    
    def cross_entropy(self, h, A, b, y, threshold=5):
        """
        calculate cross entropy loss.
        """
        s = np.dot(A, h) + b
        # to avoid overflow, approximately define `log(1+exp(s)) = s`
        if s < -threshold:
            L = y * -s
        elif s > threshold:
            L = (1 - y) * s
        else:
            p = self.sigmoid(s)
            L = -y * np.log(p) - (1 - y) * np.log(1- p)
        return L
    
    def update(self, t_hop, eps=0.001, alpha=0.0001):
        """
        backward gradient to update params.
        minimize Loss on each params; W, A, b.
        """
        # For A
        h = self.forward(t_hop, self.W, self.X)
        A_deriv = []
        for i in range(len(self.A)):
            A_delta = self.A.copy()
            A_delta[i] += eps
            A_deriv.append((self.cross_entropy(h, A_delta, self.b, self.y) - self.cross_entropy(h, self.A, self.b, self.y))/eps)
            
        # For b
        b_deriv = (self.cross_entropy(h, self.A, self.b + eps, self.y) - self.cross_entropy(h, self.A, self.b, self.y))/eps

        # For W
        W_deriv = []
        for i in range(len(self.W)):
            temp = []
            for j in range(len(self.W[i])):
                W_delta = self.W.copy()
                W_delta[i][j] += eps
                temp.append((self.cross_entropy(self.forward(t_hop, W_delta, self.X), self.A, self.b, self.y) - self.cross_entropy(self.forward(t_hop, self.W, self.X), self.A, self.b, self.y))/eps)
            W_deriv.append(temp)
            
        # Update all parameters
        A_new = self.A - np.dot(alpha, A_deriv)
        b_new = self.b - alpha * b_deriv
        W_new = self.W - np.dot(alpha, W_deriv)

        return np.array(A_new), np.array(b_new), np.array(W_new)
    
    def train(self, t_hop=2, epoch=60, eps=0.001, alpha=0.0001):
        """
        trainer. you can change iteration num.
        """
        # start iteration
        for i in range(epoch):
            # update params
            self.A, self.b, self.W = self.update(t_hop=t_hop, eps=eps, alpha=alpha)
            # calculate and print loss of a epoch to console
            loss = self.cross_entropy(self.forward(t_hop, self.W, self.X), self.A, self.b, self.y)
            print('epoch: {} loss: {}'.format(i + 1, loss))
    
    def predict(self, X, t_hop=2):
        h = self.forward(t_hop, self.W, X)
        s = np.dot(self.A, h) + self.b
        p = 1. / (1. + np.exp(-s.astype(float)))
        return int(p > 0.5)