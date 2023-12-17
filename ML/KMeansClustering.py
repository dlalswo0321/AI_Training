import numpy as np
import matplotlib.pyplot as plt

fruits = np.load("hg-mldl-master\\fruits_300.npy")
print(fruits.shape)

'''
plt.imshow(fruits[0], cmap = 'gray'
plt.show()
plt.imshow(fruits[0], cmap = 'gray_r')
plt.show()
'''

# 정답을 이미 알고 있는 경우
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 평균값과 가까운 사진 고르기
apple_mean = np.average(apple, axis = 0).reshape(100,100)
abs_diff = np.abs(fruits - apple_mean) # abs_diff = (300, 100, 100)
abs_mean = np.mean(abs_diff, axis=(1,2)) # 차원이 두 방향으로 축소됨
print(abs_mean.shape) # abs_mean = (300,)

apple_index = np.argsort(abs_mean)[:100] # 차이 작은 순서대로 index 반환
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()

# KMeansClustering 사용
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(np.unique(km.labels_, return_counts=True))

# Cluster 성능 확인
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(fruits[km.labels_==0])
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 최적의 k 찾기
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()