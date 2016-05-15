import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import NeuralNetwork as nn
import csv
from os import listdir

data = pd.read_csv("train.csv")
X_data = data.iloc[:40000, 1:].values  # X_train(100 x 784)
X_data.astype(float)
y_data = data.iloc[:40000, 0].values  # y_train(100 x 1)
select_value = np.arange(40000)
NN = nn.NeuralNetwork(3, 1, 0.01)

for i in range(3500):
    print(i, " :")
    np.random.shuffle(select_value)
    X_train = X_data[select_value[:750]]
    y_train = y_data[select_value[:750]] 
    X_train = X_train / 256
    # print(y_train)
    NN.train(X_train, y_train)


def forImage():
    url = r"E:\Machine Learning\Handwritten digit\Python Code\DigitRecognition\image"
    file_list = listdir(url)

    for i in range(len(file_list)):
        print(file_list[i])
        image = cv2.imread(url + "\\" + file_list[i])
        cv2.imshow("image", image)
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.waitKey(0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print(gray_image.shape)
        gray_image.astype(float)
        gray_image = 255 - gray_image
        # print(gray_image)
        gray_image = gray_image / 256
        out = NN.classification(gray_image)

        pre = -1
        max = 0
        for j in range(len(out)):
            if (out[j] > max):
                max = out[j]
                pre = j
        print("-----Predict: ", pre)


def forCsv():

    data2 = pd.read_csv("test.csv")
    X = data2.iloc[:, :].values  # X_train(100 x 784)
    X.astype(float)
    X = X / 256
    with open("output.csv", "w", newline="") as mycsvfile:
        target = csv.writer(mycsvfile)
        target.writerow(["ImageId", "Label"])

        for i in range(X.shape[0]):
            print("case ", i)
            out = NN.classification(X[i])
            pre = -1
            max = 0
            for j in range(len(out)):
                if (out[j] > max):
                    max = out[j]
                    pre = j
            target.writerow([str(i + 1), str(pre)])

forImage()
