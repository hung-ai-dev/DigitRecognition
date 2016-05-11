import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import NeuralNetwork as nn

data = pd.read_csv("mnist_train.csv", header=None)
X_data = data.iloc[:2000, 1:].values  # X_train(100 x 784)
X_data.astype(float)
y_data = data.iloc[:2000, 0].values  # y_train(100 x 1)
select_value = np.arange(2000)
NN = nn.NeuralNetwork(3, 1000, 0.01)

for i in range(1):
    np.random.shuffle(select_value)
    X_train = X_data[select_value[:100]]
    y_train = y_data[select_value[:100]]
    X_train = X_train / 256
    # print(y_train)
    NN.train(X_train, y_train)

image = cv2.imread('5_1.jpg')
image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("image", image)
cv2.waitKey(0)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(gray_image.shape)
print(gray_image)
gray_image.astype(float)
gray_image = gray_image / 256
NN.classification(gray_image)

image = cv2.imread('4_5.jpg')
image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("image", image)
cv2.waitKey(0)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(gray_image.shape)
print(gray_image)
gray_image.astype(float)
gray_image = gray_image / 256
NN.classification(gray_image)


"""
data2 = pd.read_csv("mnist_test.csv", header = None)
X = data2.iloc[:, 1:].values  # X_train(100 x 784)
X.astype(float)
y = data2.iloc[:, 0].values  # y_train(100 x 1)
X = X / 256
y = NN.makeOutput(y)

for i in range(100):
    print("case ", i)
    NN.classification(X[i])
    print(y[i])


"""