#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn as sk
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS


# In[ ]:


# Générer 5 clusters gaussiens qui contiennent 500 points bidimensionnels au total en utilisant la fonction sklearn.dataset.make_blobs.

centers_random = np.random.randint(3,6)
n_samples_random = np.random.randint(100, 1000)
X, y = make_blobs(n_samples=n_samples_random, centers=centers_random, n_features=2)
plt.scatter(X[:,0], X[:,1])
plt.show()


# In[ ]:


def initialize_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


# In[ ]:


def sse_distance(x, y):
    return np.sum((x - y) ** 2)


# In[ ]:


def find_closest_center(centers, x):
    distances = [sse_distance(x, center) for center in centers]
    return np.argmin(distances)


# In[ ]:


def compute_clusters(X, centers):
    labels = [find_closest_center(centers, x) for x in X]
    clusters = [[] for _ in range(len(centers))]
    for i, label in enumerate(labels):
        clusters[label].append(X[i])
    return clusters


# In[ ]:


def sse_error(X, centers):
    labels = [find_closest_center(centers, x) for x in X]
    distances = [sse_distance(X[i], centers[labels[i]]) for i in range(len(X))]
    return np.sum(distances)
    


# In[ ]:


def initialize_centers(X, k, method='random'):
    if method == 'random':
        indices = np.random.choice(X.shape[0], size=k, replace=False)
        return X[indices]
    elif method == 'kmeans++':
        centers = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, k):
            dist = np.min([np.linalg.norm(X - center, axis=1) for center in centers], axis=0)
            probs = dist / np.sum(dist)
            next_center = X[np.random.choice(X.shape[0], p=probs)]
            centers.append(next_center)
        return np.array(centers)
    else:
        raise ValueError(f"Unknown initialization method {method}")


# In[ ]:


def visualize_clusters(points, labels, centers):
    colors = list(CSS4_COLORS.values())
    for i, color in enumerate(colors[:len(centers)]):
        mask = labels == i
        plt.scatter(points[mask, 0], points[mask, 1], color=color, label=f"Cluster {i}")
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=200, label='Centers')
    plt.legend()
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_clusters(points, labels, centers, filename):
    # Utilisez une colormap pour obtenir plus de couleurs
    colors = plt.cm.tab20(np.linspace(0, 1, len(centers)))

    for i, color in enumerate(colors):
        mask = labels == i
        plt.scatter(points[mask, 0], points[mask, 1], color=color, label=f"Cluster {i}")
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=200, label='Centers')
    plt.legend()

    # Sauvegarder le tracé dans un fichier et fermer le graphique
    plt.savefig(filename)
    plt.close()

    return filename

    


# In[ ]:


def kmeans(X, k, num_it=16, init='random'):
    centers = initialize_centers(X, k, method=init)
    history = {"centers": [], "sse_error": [], "clusters": [], "visualize": []}
    tps_convergence = num_it
    for i in range(num_it):
        clusters = compute_clusters(X, centers)
        history["clusters"].append(clusters)
        new_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        sse = sse_error(X, new_centers)
        history["centers"].append(centers)
        history["sse_error"].append(sse)
        
        # Générer un nom de fichier unique pour chaque graphique
        filename = f"clusters_iteration_{i}.png"
        visualize_clusters(X, np.array([find_closest_center(centers, x) for x in X]), np.array(centers), filename)
        history["visualize"].append(filename)
        
        if np.allclose(centers, new_centers):
            tps_convergence = i
            break 
        centers = new_centers
    return history, tps_convergence


# In[ ]:


from IPython.display import Image
# Test de la fonction visualize_clusters
history, tps_convergence = kmeans(X, centers_random, 10)
historypp, tps_convergencepp = kmeans(X, centers_random, 10,init='kmeans++')
print(centers_random)
plt.scatter(X[:,0], X[:,1])
plt.show()
# visulaize les clusters
for i, filename in enumerate(history["visualize"]):
    print(f"Iteration {i}")
    display(Image(filename=filename))


# In[ ]:


from sklearn.metrics import pairwise_distances_argmin_min
for i in range(1, 3):
    # Exécuter k-means et k-means++
    history,temps_convergence = kmeans(X, centers_random,5, init='random')
    historypp,temps_convergencepp = kmeans(X,centers_random,5, init='kmeans++')

    # Comparer les résultats
    print(f"k-means: {temps_convergence} iterations, SSE = {history['sse_error'][-1]}")
    print(f"k-means++: {temps_convergencepp} iterations, SSE = {historypp['sse_error'][-1]}")

    # Sotcker les résultats de sse_error
    error = []
    error.append(historypp['sse_error'][-1] - history['sse_error'][-1]) 
    # Afficher les derniers graphiques de chauqe algorithme
    print("k-means")
    display(Image(filename=history["visualize"][-1]))
    print("k-means++")
    display(Image(filename=historypp["visualize"][-1]))
# Faire le somme des erreurs
print(np.sum(error))


# In[ ]:


def mini_batch_kmeans(X, k, batch_size=100, max_iters=100):
    # Initialiser les centres de manière aléatoire
    centers = X[np.random.choice(X.shape[0], size=k, replace=False)]

    for _ in range(max_iters):
        # Sélectionner un sous-ensemble aléatoire des données
        batch = X[np.random.choice(X.shape[0], size=batch_size, replace=False)]

        # Attribuer chaque point du sous-ensemble au centre le plus proche
        labels = np.argmin(np.linalg.norm(batch[:, np.newaxis] - centers, axis=2), axis=1)

        # Mettre à jour les centres
        for i in range(k):
            if np.any(labels == i):
                centers[i] = np.mean(batch[labels == i], axis=0)

    # Attribuer chaque point à son centre le plus proche
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

    return centers, labels


# In[ ]:


# Générer un grand échantillon de données
centers_random = 10
X, _ = make_blobs(n_samples=10000, centers=centers_random, n_features=2)

# Exécuter Mini-Batch K-Means
centers, labels = mini_batch_kmeans(X, k=centers_random)

# Visualiser les résultats
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200)
plt.show()


# In[ ]:


# Comparaison de kmeans++ et du kmeans de skit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Générer un grand échantillon de données
centers_random = 10
X, _ = make_blobs(n_samples=7000, centers=centers_random, n_features=2)

# Exécuter k-means et k-means++
historypp, temps_convergencepp = kmeans(X, centers_random, 10, init='kmeans++')

# Créer une instance de KMeans
kmeanssklearn = KMeans(n_clusters=centers_random, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Adapter le modèle aux données
kmeanssklearn.fit(X)

# Obtenir les labels des clusters pour chaque point de données
labels = kmeanssklearn.labels_

# Obtenir les centres des clusters
centers = kmeanssklearn.cluster_centers_

# Comparer les résultats
print(f"k-means++: {temps_convergencepp} iterations, SSE = {historypp['sse_error'][-1]}")
print(f"k-means sklearn: {kmeanssklearn.n_iter_} iterations, SSE = {kmeanssklearn.inertia_}")

# Afficher les derniers graphiques de chauqe algorithme
print("k-means++")
display(Image(filename=historypp["visualize"][-1]))
print("k-means sklearn")
visualize_clusters(X, labels, centers, "kmeanspp.png")
display(Image(filename="kmeanspp.png"))


