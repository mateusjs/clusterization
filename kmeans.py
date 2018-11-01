import numpy as np
'''
Kmeans algorithm
'''
class Kmeans(object):
  '''
  k = number of centroids
  n_iter = number of iterations
  '''
  def __init__(self, k=3, n_iter=20):
    self.k = k
    self.iterations = n_iter

  def fit(self, X, Y=None):
    self.centroids = np.random.uniform(
      low=np.amin(X, axis=0),
      high=np.amax(X, axis=0),
      size=(self.k, X.shape[1])
    ) # define centroides aleatórios
    old_centroids = np.array([])

    for _ in range(self.iterations):
      if (np.array_equal(old_centroids, self.centroids)): # centroides não mudaram, para o processo
        break
      
      old_centroids = np.copy(self.centroids)

      labels = np.apply_along_axis( # calcula a distância de cada centroide, pega o índice do centroide mais perto
        lambda instancia, centroides: np.argmin(np.sum(np.power(instancia - centroides, 2), axis=1)),
        1,
        X,
        self.centroids
      )

      for index in range(self.k): # recalcula os centroides
          self.centroids[index] = np.mean(X[labels == index])

    return labels
