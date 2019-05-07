import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import *
# from util import *
from model import GNN

# create datasets
X, y = load_train()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train
gnn = GNN()
# optimizer = 'SGD'
# optimizer = 'Momentum'
optimizer = 'Adam'
gnn.train(X_train, y_train, epoch=10, minibatch=128, optimizer=optimizer)

# test
y_hat = gnn.predict(X_test)
correct = 0
for i in range(len(y_test)):
    if y_hat[i] == y_test[i]:
        correct += 1
print("acc: ", correct / len(y_test))

# predict
X_submit = load_test()
pred = gnn.predict(X_submit)
with open('../prediction.txt', 'w') as f:
    for p in pred:
        f.write(str(p))
        f.write('\n')

# clear memory
del gnn, X, y