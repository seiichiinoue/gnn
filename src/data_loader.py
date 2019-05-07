import numpy as np

def load_train(num=2000):
    X, y = [], []
    for i in range(num):
        with open('../inputs/train/{}_graph.txt'.format(str(i))) as f:
            n = int(f.readline().strip())
            graph = np.array([list(map(int, f.readline().strip().split())) for _ in range(n)])
        with open('../inputs/train/{}_label.txt'.format(str(i))) as f:
            label = int(f.readline().strip())
        X.append(graph)
        y.append(label)
    return np.array(X), np.array(y)

def load_test():
    X = []
    for i in range(500):
        with open('../inputs/train/{}_graph.txt'.format(str(i))) as f:
            n = int(f.readline().strip())
            graph = np.array([list(map(int, f.readline().strip().split())) for _ in range(n)])
        X.append(graph)
    return np.array(X)