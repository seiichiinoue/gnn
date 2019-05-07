import numpy as np
from util import *

class GNN(object):
    def __init__(self, embed_dim=8, t_hop=2):
        """
        Args:
            embed_dim : embedding dimention
            t_hop : num of aggregation
        """
        self.embed_dim = embed_dim
        self.t_hop = t_hop
        # initialize parameters
        self.W = np.random.normal(0, 0.4, (self.embed_dim, self.embed_dim))
        self.A = np.random.normal(0, 0.4, self.embed_dim)
        self.b = 0
        # # optimizer
        # self.optimizer = optimizer

    def forward(self, adj):
        """
        Args:
            adj : inputs; adjacent matrix of target graph
        Return:
            s : unactivated output value
        """
        # aggregate layer
        h = Aggregate_t_times(self.W, adj, embed_dim=self.embed_dim, t_hop=self.t_hop)
        # fc layer
        s = Linear(self.A, h, self.b)
        return s
    
    def calc_loss(self, s, y):
        """
        Args:
            s : unactivated value
            y : target label
        Return:
            loss
        """
        loss = cross_entropy_enhanced(s, y)
        return loss
    
    
    def calc_grad(self, X, y, eps=0.001, alpha=0.0001):
        """
        calculate gradient for each params.
        Args:
            X : adjacent matrix
            y : target label
            eps : minimum value
            alpha : learning rate
        Return:
            gradient of each params.
        """

        h = Aggregate_t_times(self.W, X, embed_dim=self.embed_dim, t_hop=self.t_hop)
    
        # For A
        A_grad = []
        for i in range(len(self.A)):
            A_delta = self.A.copy()
            A_delta[i] += eps
            A_grad.append((self.calc_loss(Linear(A_delta, h, self.b), y) - self.calc_loss(Linear(self.A, h, self.b), y)) / eps)
    
        # For b
        b_grad = (self.calc_loss(Linear(self.A, h, self.b + eps), y) - self.calc_loss(Linear(self.A, h, self.b), y)) / eps

        # For W
        W_grad = []
        for i in range(len(self.W)):
            tmp = []
            for j in range(len(self.W[i])):
                W_delta = self.W.copy()
                W_delta[i][j] += eps
                h_ = Aggregate_t_times(W_delta, X, embed_dim=self.embed_dim, t_hop=self.t_hop)
                tmp.append((self.calc_loss(Linear(self.A, h_, self.b), y) - self.calc_loss(Linear(self.A, h, self.b), y)) / eps)
            W_grad.append(tmp)

        return np.array(A_grad, dtype='float64'), b_grad.astype('float64'), np.array(W_grad, dtype='float64')

    def train_loader(self, X, y, minibatch=128):
        idx = np.random.permutation(len(y))
        mini_X, mini_y = [], []
        for i in range(len(idx)):
            mini_X.append(X[idx[i]])
            mini_y.append(y[idx[i]])
            if len(mini_y) == minibatch:
                yield np.array(mini_X), np.array(mini_y)
                mini_X, mini_y = [], []
    
    def train(self, X, y, minibatch=128, epoch=60, eps=0.001, alpha=0.0001, optimizer='SGD', **kwargs):
        # decide optimizer
        if optimizer == 'SGD':
            opt = SGD(lr=alpha)
        elif optimizer == 'Momentum':
            opt = Momentum(lr=alpha, momentum=0.9)
        elif optimizer == 'Adam':
            opt = Adam(lr=alpha, beta1=0.9, beta2=0.999)
        # start iteration
        for i in range(epoch):
            running_loss = 0 
            for X_b, y_b in self.train_loader(X, y, minibatch=minibatch):
                # pool params
                grads = [np.zeros_like(self.A), 0, np.zeros_like(self.W)]
                # A_pool, b_pool, W_pool= np.zeros_like(self.A), 0, np.zeros_like(self.W)
                for j in range(minibatch):
                    # calculate gradient
                    grads_new = self.calc_grad(X_b[j], y_b[j], eps=eps, alpha=alpha)
                    # pool grads
                    for k in range(len(grads)):
                        grads[k] += grads_new[k]
                    # calculate loss
                    h_hat = Aggregate_t_times(self.W, X_b[j], embed_dim=self.embed_dim, t_hop=self.t_hop)
                    running_loss += self.calc_loss(Linear(self.A, h_hat, self.b), y_b[j])
                # calculate mean of grads
                for k in range(len(grads)):
                    grads[k] /= minibatch
                # update params
                opt.update([self.A, self.b, self.W], grads)
                
            print('epoch: {} loss: {}'.format(i + 1, running_loss / len(y)))
            # print log
            with open('./log.txt', 'a') as f:
                print("epoch: %.d running_loss: %.10f " % (i+1, running_loss), file=f)
            running_loss = 0

    def predict(self, X):
        """
        Args:
            X : adjacent matrixes
        Return:
            binary : infered labels
        """
        pred = []
        for adj in X:
            p = Sigmoid(self.forward(adj))
            pred.append(int(p > 0.5))
        return np.array(pred)


