import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, nLayers, nIteration=10, learningRate=0.1, regularization=0.1):
        ### nLayers: number of layers
        ### nIteration: number of iterations
        self.nLayers = nLayers
        self.nIteration = nIteration
        self.learningRate = learningRate
        self.regularization = regularization

    def init(self, X_train, y_train):
        ### self.nLayer: number of neural at each layer (28*28, 30, 10)
        ### self.value: value of neural at each layer
        ### self.w: Theta (30, 28*28 + 1 bias unit) and (10, 30 + 1 bias unit)
        ### X_train: add 1 more bias column
        ### y_train

        self.nLayer = [int(X_train.shape[1]), 30, 10]

        self.m = X_train.shape[0]
        self.X_train = np.append(np.ones((self.m, 1)), X_train, axis=1)
        self.y_train = y_train
        self.y_train = self.makeOutput()

        self.value = [
            np.zeros((self.m, self.nLayer[0] + 1)),
            np.zeros((self.m, self.nLayer[1] + 1)),
            np.zeros((self.m, self.nLayer[2]))
        ]

        self.errorLocal = [
            np.zeros((self.m, self.nLayer[1])),
            np.zeros((self.m, self.nLayer[2]))
        ]

        self.w = [
            np.random.uniform(-0.12, 0.12, (self.nLayer[1], self.nLayer[0] + 1)),
            np.random.uniform(-0.12, 0.12, (self.nLayer[2], self.nLayer[1] + 1))
        ]

        self.cost = 0

    def makeOutput(self):
        out = np.zeros((y_train.shape[0], 10))
        for i in range(y_train.shape[0]):
            out[i, y_train[i]] = 1
        return out

    def feedForward(self):
        self.value[0] = self.X_train
        self.value[1] = self.sigmoid(self.value[0].dot(self.w[0].T))
        self.value[1] = np.append(np.ones((self.m, 1)), self.value[1], axis=1)
        self.value[2] = self.sigmoid(self.value[1].dot(self.w[1].T))

    ### back propagation
    def backPropa(self):
        self.errorLocal[1] = self.value[2] - self.y_train
        self.errorLocal[0] = self.errorLocal[1].dot(self.w[1][:, 1:]) * self.sigmoid_derivative(
            self.value[0].dot(self.w[0].T))

        self.w[1] -= self.learningRate * self.errorLocal[1].T.dot(self.value[1])
        self.w[0] -= self.learningRate * self.errorLocal[0].T.dot(self.X_train)

        # add regularization
        self.w[1][:, 1:] -= self.regularization / self.m * self.w[1][:, 1:]
        self.w[0][:, 1:] -= self.regularization / self.m * self.w[0][:, 1:]

    def train(self, X_train, y_train):
        self.init(X_train=X_train, y_train=y_train)

        for i in range(0, self.nIteration):
            print(i, ':')
            self.feedForward()
            self.backPropa()
            print('Cost = ', self.costFunction())

    def costFunction(self):
        # not add regularization yet
        J = np.log(self.value[2]) * self.y_train + np.log(1 - self.value[2]) * (1 - self.y_train)
        J = -np.sum(J) / self.m

        # add regularization
        J += self.regularization * (np.sum(self.w[0][:, 1:] ** 2) + np.sum(self.w[1][:, 1:] ** 2)) / (2 * self.m)

        return J

    def classification(self, X_test):
        X_test = np.insert(X_test, 0, 1.0)
        hidden = self.sigmoid(X_test.dot(self.w[0].T))
        hidden = np.insert(hidden, 0, 1.0)
        out = self.sigmoid(hidden.dot(self.w[1].T))
        out = np.round(out)
        print(out)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sigVal = self.sigmoid(z)
        return sigVal * (1 - sigVal)


data = pd.read_csv("mnist_train.csv", header=None)
X_data = data.iloc[:2000, 1:].values  # X_train(100 x 784)
X_data.astype(float)
y_data = data.iloc[:2000, 0].values  # y_train(100 x 1)

X_train = X_data[:6000]
y_train = y_data[:6000]

### vẽ lại
"""
ig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""

X_train = X_train / 256
NN = NeuralNetwork(nLayers=3, nIteration=10000, learningRate=0.0001)
# print(y_train)
NN.train(X_train, y_train)

"""
testData = pd.read_csv("mnist_test_10.csv", header=None)
X_test = testData.iloc[:, 1:].values  # X_train(100 x 784)
X_test.astype(float)
y_test = testData.iloc[:, 0].values  # y_train(100 x 1)

X_test = X_test / 256
y_out = NN.makeOutput(y_data)

for i in range(50):
    NN.classification(X_data[i + 1010], y_out[i + 1010])
"""

image = cv2.imread('1.jpg')
image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("image", image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image.astype(float)
gray_image = gray_image / 256
print(gray_image)
NN.classification(gray_image)
