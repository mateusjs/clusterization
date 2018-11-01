import numpy as np

'''
DBSCAN algorithm
'''


class DbScan(object):
    '''
    epsilon = maximum distance between neighbors
    n_neighbors = Minimum nunber of neighbors for a point to define a cluster
    '''

    def __init__(self, epsilon, n_neighbors=5):
        self.epsilon = epsilon * epsilon
        self.n_neighbors = n_neighbors

    def fit(self, X):
        classes = np.full(X.shape[0], -1, dtype=np.int)
        class_id = 0

        for index in range(X.shape[0]):
            if classes[index] != -1: # já foi calculado
                continue

            indexes_bool = np.sum(np.square(np.subtract(X, X[index])), axis=1) <= self.epsilon # True pros vizinhos

            if np.sum(indexes_bool) < self.n_neighbors: # não é um core-point
                continue

            lista_de_indices = np.argwhere(indexes_bool).flatten() # indice dos vizinhos
            iterator = 1 # pula o recálculo do próprio elemento

            classes[index] = class_id # define o cluster
            class_id += 1

            while iterator < len(lista_de_indices):
                indexes_bool = np.sum(np.square(np.subtract(X, X[lista_de_indices[iterator]])), axis=1) <= self.epsilon # booleanos

                classes[lista_de_indices[iterator]] = class_id # muda a classe do vizinho

                iterator += 1

                if np.sum(indexes_bool) < self.n_neighbors: # não é um cluster
                    continue

                # adiciona os novos vizinhos do cluster
                indexes = np.argwhere(indexes_bool).flatten() # indices
                lista_de_indices = np.append(lista_de_indices, np.setdiff1d(indexes, lista_de_indices))
                #######################################
        return classes