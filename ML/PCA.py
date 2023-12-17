import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50) # n_components : 주성분 수(차원을 얼마나 줄일지)
pca.fit(fruits_2d)

print(pca.components_.shape)

import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(pca.components_.reshape(-1, 100, 100))

print(fruits_2d.shape) # (300, 10000)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # (300, 50)

# Data reconstruct
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape) # (300, 10000)
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
    
# Explained variance
print(np.sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)

# Logistic regression w/PCA
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

target = np.array([0] * 100 + [1] * 100 + [2] * 100)

from sklearn.model_selection import cross_validate

print("[ w/o pca ]")
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

print("[ w/ pca ]")
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

pca = PCA(n_components = 0.5) # explained variance의 50%에 달하는 PC를 찾도록
pca.fit(fruits_2d)
print(pca.n_components_)
fruits_pca = pca.transform(fruits_2d)

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# K-Means clustering w/PCA
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()