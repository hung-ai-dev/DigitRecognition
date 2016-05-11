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
NN = nn.NeuralNetwork(3, 1000, 0.0001)

for i in range(500):
    np.random.shuffle(select_value)
    X_train = X_data[select_value[:1000]]
    y_train = y_data[select_value[:1000]]
    X_train = X_train / 256
    # print(y_train)
    NN.train(X_train, y_train)


image = cv2.imread('1.jpg')
image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("image", image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image.astype(float)
gray_image = gray_image / 256
print(gray_image)
NN.classification(gray_image)

