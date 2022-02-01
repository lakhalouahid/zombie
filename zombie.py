#!/usr/bin/env python
import numpy as np
import numpy as cp
import pickle
import argparse
from sklearn.datasets import make_classification, make_moons, make_circles

train = False

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=1, type=int, help="Help text")
parser.add_argument("--x1", default=50, type=float, help="Help text")
parser.add_argument("--x2", default=50, type=float, help="Help text")
args = parser.parse_args()
x1 = args.x1
x2 = args.x2
dataset = args.dataset

def sigmoid(x):
    return 1/(1+cp.exp(-x))



class ZombieNN():
    def __init__(self, dataset_idx=0):
        self.W1 = cp.random.randn(2, 100) * 0.05
        self.W2 = cp.random.randn(100, 1000) * 0.001
        self.W3 = cp.random.randn(1000, 100) * 0.001
        self.W4 = cp.random.randn(100, 1) * 0.1
        self.b1 = cp.zeros((1, 100))
        self.b2 = cp.zeros((1, 1000))
        self.b3 = cp.zeros((1, 100))
        self.b4 = cp.zeros((1, 1))
        self.range_X = []
        self.dataset_idx = dataset_idx


    def predict(self, X, range_X=[[0, 100], [0, 100]]):
        X[0] = self.range_X[0][0]  + (self.range_X[0][1]-self.range_X[0][0])*(X[0]-range_X[0][0])/range_X[0][1]
        X[1] = self.range_X[1][0]  + (self.range_X[1][1]-self.range_X[1][0])*(X[1]-range_X[1][0])/range_X[1][1]
        X = cp.array(X).reshape(1, 2)
        h2 = X @ self.W1 + self.b1
        z2 = h2*(h2>=0)
        h3 = z2 @ self.W2 + self.b2
        z3 = h3*(h3>=0)
        h4 = z3 @ self.W3 + self.b3
        z4 = h4*(h4>=0)
        h5 = z4 @ self.W4 + self.b4
        z5 = sigmoid(h5)
        return float(z5[0][0])

    def train(self, X, y, lr=0.1, batch=1000, epoch=100, mu=0.9, reg=0.001):
        X = cp.array(X)
        y = cp.array(y)
        max_X = cp.max(X, axis=0)
        min_X = cp.min(X, axis=0)
        self.range_X.append([min_X[0], max_X[0]])
        self.range_X.append([min_X[1], max_X[1]])
        y = y.reshape(-1, 1)
        m = y.shape[0]
        loss = -cp.log(1/6)
        mW1, mW2, mW3, mW4, mb1, mb2, mb3, mb4 = 0, 0, 0, 0, 0, 0, 0, 0 
        for p in range(epoch):
            print(f"epoch: {p+1}, loss: {loss}--------------------------------------")
            for i in range(int(cp.ceil(m/batch))):
                # forward
                rg = cp.arange(batch*i, min(batch*(i+1), m), dtype=cp.int32)
                h2 = X[rg, :] @ self.W1 + self.b1
                r2 = (h2>=0)
                z2 = h2*r2
                h3 = z2 @ self.W2 + self.b2
                r3 = (h3>=0)
                z3 = h3*r3
                h4 = z3 @ self.W3 + self.b3
                r4 = (h4>=0)
                z4 = h4*r4
                h5 = z4 @ self.W4 + self.b4
                z5 = sigmoid(h5)
                # loss
                loss = - cp.mean(y[rg]*cp.log(z5) + (1-y[rg])*cp.log(1 - z5)) + \
                        0.5 * reg * (cp.mean(self.W1**2)+cp.mean(self.W2**2)+cp.mean(self.W3**2)+cp.mean(self.W4**2))

                # backprop
                dy = -(y[rg]/z5 - (1 - y[rg])/(1 - z5))/batch
                dh5 = dy * z5 * (1 - z5) # sigmoid
                dW4 = z4.T @ dh5 + reg * self.W4/(self.W4.shape[0]*self.W4.shape[1])
                db4 = cp.sum(dh5, axis=0)
                dz4 = dh5 @ self.W4.T
                dh4 = dz4 * r4
                dW3 = z3.T @ dh4 + reg * self.W3/(self.W3.shape[0]*self.W3.shape[1])
                db3 = cp.sum(dh4, axis=0)
                dz3 = dh4 @ self.W3.T
                dh3 = dz3 * r3
                dW2 = z2.T @ dh3 + reg * self.W2/(self.W2.shape[0]*self.W2.shape[1])
                db2 = cp.sum(dh3, axis=0)
                dz2 = dh3 @ self.W2.T
                dh2 = dz2 * r2
                dW1 = X[rg].T @ dh2 + reg * self.W1/(self.W1.shape[0]*self.W1.shape[1])
                db1 = cp.sum(dh2, axis=0)

                for mem, grad, param in zip([mW1, mW2, mW3, mW4, mb1, mb2, mb3, mb4],
                        [dW1, dW2, dW3, dW4, db1, db2, db3, db4],
                        [self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4]):
                    mem = mu * mem - lr * grad
                    param += mem
        self.save(f"zombie_dataset_{self.dataset_idx}.bin")

    def save(self, filename='zombie.bin'):
        with open(filename, 'bw') as fd:
            pickle.dump(self, fd)

def load_zoombie(filename='zombie.bin'):
    with open(filename, 'br') as fd:
        return pickle.load(fd)


def load_datasets(n_samples):
    X, y = make_classification(n_samples=n_samples,
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        make_moons(n_samples=n_samples, noise=0.3, random_state=0),
        make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]
    return datasets

def main():
    # training
    if train:
        datasets = load_datasets(10000)
        zombie_dataset1 = ZombieNN(1)
        zombie_dataset2 = ZombieNN(2)
        zombie_dataset3 = ZombieNN(3)
        zombie_dataset1.train(datasets[0][0], datasets[0][1], epoch=200, batch=100)
        zombie_dataset2.train(datasets[1][0], datasets[1][1], epoch=200, batch=100)
        zombie_dataset3.train(datasets[2][0], datasets[2][1], epoch=200, batch=100)
    else:
        datasets = load_datasets(10000)
        zombie_dataset = load_zoombie(f'./zombie_dataset_{dataset}.bin')
        print(zombie_dataset.predict([x1, x2]), end="")



if __name__ == '__main__':
    main()
