# %% [markdown]
# Importation des librairies de Python nécessaires.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA

# %% [markdown]
# Chargement des données ou bien simuler aléatoirement les données de la dimension
# supérieur à 4.
# 

# %%
X, y = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)
print(f"Taille des données: {X.shape}")

# %% [markdown]
# Visualisez et donner la taille des données

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.xlabel('Premier axe')
plt.ylabel('Deuxième axe')
plt.title('Visualisation du data')
plt.show()

# %% [markdown]
# Implémentez l’algorithme K-moyenne

# %% [markdown]
# Aléatoire

# %%
kmeans_random = KMeans(n_clusters=4, init='random', n_init=10, random_state=42)
kmeans_random.fit(X)
labels_random = kmeans_random.labels_
centers_random = kmeans_random.cluster_centers_
print("Aléatoire clusters Initialisation")
print(centers_random)

# %% [markdown]
# K-means++

# %%
kmeans_plus = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans_plus.fit(X)
labels_plus = kmeans_plus.labels_
centers_plus = kmeans_plus.cluster_centers_
print("K-means++ clusters Initialisation ")
print(centers_plus)

# %% [markdown]
# Visualisation avec les deux stratégies

# %%
centers_plus_pca = pca.transform(centers_plus)
centers_random_pca = pca.transform(centers_random)

# %% [markdown]
# Visualisation aléatoire

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_random, palette='viridis')
plt.scatter(centers_random_pca[:, 0], centers_random_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe')
plt.ylabel('Deuxième axe')
plt.title('Aléatoire')
plt.show()

# %% [markdown]
# Visualisation K-means++

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_plus, palette='viridis')
plt.scatter(centers_plus_pca[:, 0], centers_plus_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe')
plt.ylabel('Deuxième axe')
plt.title('K-means++')
plt.show()

# %% [markdown]
# méthodes de validation de Clustering.

# %%
# Silhouette score
silhouette_avg_random = silhouette_score(X, labels_random)
silhouette_avg_plus = silhouette_score(X, labels_plus)
# Calinski-Harabasz index
calinski_harabasz_random = calinski_harabasz_score(X, labels_random)
calinski_harabasz_plus = calinski_harabasz_score(X, labels_plus)

# %% [markdown]
# Interprétez les résultats

# %%
print(f'Silhouette Score (Random): {silhouette_avg_random:.3f}')
print(f'Silhouette Score (K-means++): {silhouette_avg_plus:.3f}')
print(f'Calinski-Harabasz Index (Random): {calinski_harabasz_random:.3f}')
print(f'Calinski-Harabasz Index (K-means++): {calinski_harabasz_plus :.3f}')

# %% [markdown]
# Basé sur les métriques de validation obtrenues, les deux modèles de clustering K-Means avec initialisation aléatoire et initialisation K-means++ ont la même performance :
# 
#     Silhouette Score (Initialisation aléatoire) : 0.747
#     Silhouette Score (Initialisation K-means++) : 0.747
#     Indice de Calinski-Harabasz (Initialisation aléatoire) : 3715.491
#     Indice de Calinski-Harabasz (Initialisation K-means++) : 3715.491

# %% [markdown]
# le meilleur modèle de Clustering

# %% [markdown]
# 
# 
# Puisque les scores de validation sont identiques pour les deux modèles, il n'y a pas de "meilleur" modèle clair basé sur les informations données. Les stratégies d'initialisation aléatoire et d'initialisation K-means++ ont toutes deux donné des résultats de clustering également bons pour ce jeu de données.

# %% [markdown]
# Peut on représenter les données avec les poids des centres obtenus ?

# %% [markdown]
# Oui, il est possible de représenter les données avec les poids des centres obtenus par l'algorithme K-Means. Cette représentation peut être utile pour visualiser la répartition des données par rapport aux centroïdes de chaque cluster.

# %% [markdown]
# l’analyse en composantes principales

# %% [markdown]
# nouvelle matrice des observations.

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca

# %% [markdown]
# les valeurs propres et les vecteurs propres associes aux axes principaux.

# %%
eigenvectors = pca.components_
print("Vecteurs propres :")
print(eigenvectors)

eigenvalues = pca.explained_variance_
print("Valeurs propres :")
print(eigenvalues)

# %% [markdown]
# l’inertie de chaque axe.

# %%
explained_variance_ratio = pca.explained_variance_ratio_
print("l’inertie de chaque axe :")
print(explained_variance_ratio)

# %% [markdown]
# Vérifier que la somme des inerties de chaque axe égal la dimension de la base de
# données.

# %%
sum_explained_variance_ratio = np.sum(explained_variance_ratio)
print(f"Somme inerties : {sum_explained_variance_ratio:.3f}")

# %% [markdown]
# Représenter les données ainsi que les centres obtenus

# %%
kmeans_plus = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans_plus.fit(X)
labels_plus = kmeans_plus.labels_
centers_plus = kmeans_plus.cluster_centers_
centers_plus_pca = pca.transform(centers_plus)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_plus, palette='viridis')
plt.scatter(centers_plus_pca[:, 0], centers_plus_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe')
plt.ylabel('Deuxième axe')
plt.title('K-means++ Cluster')
plt.show()

# %% [markdown]
#  Interprétation des résultats de l'analyse en composantes principales (PCA) 

# %% [markdown]
# 
# 
#     La PCA a permis de réduire la dimensionnalité des données tout en conservant une part importante de l'inertie (variance) initiale.
#     L'analyse des valeurs propres et des vecteurs propres associés aux axes principaux fournit des informations sur les directions de plus grande variance dans les données.
#     La représentation des données et des centres des clusters sur les deux premiers axes principaux permet de visualiser la structure et la répartition des groupes dans cet espace de plus faible dimension.
#     Cette visualisation PCA complémentaire aux résultats du clustering K-Means peut aider à mieux comprendre la structure sous-jacente des données et la pertinence des groupes identifiés.
# 
# En résumé, l'interprétation conjointe des résultats du clustering K-Means et de l'analyse PCA permet d'avoir une compréhension plus approfondie de la structure et des caractéristiques des données étudiées.


