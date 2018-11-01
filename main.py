import pandas as pd
from kmeans import Kmeans
from dbscan import DbScan
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dataset = pd.read_csv('banana.csv', sep=',', header=None)
x = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 2].values - 1

del dataset

scan = DbScan(.1, 5)

classes = scan.fit(x)

plt.scatter(x[:, 0], x[:, 1], s=2, c=classes)
plt.show()

plt.scatter(x[:, 0], x[:, 1], s=2, c=y)
plt.show()