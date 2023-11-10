# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

3.Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4.Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7.Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements
## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Mythili D
RegisterNumber:  212222040104
```
```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers (1).csv')

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:
## df.head():
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/d18736b7-334d-44e7-aba2-efc469020bce)
## df.info():
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/c584fcba-2cf7-4b00-a054-047530a7c563)
## data.isnull().sum():
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/5ffb4453-7047-48ca-8d00-bb430f1cad7e)
## Elbow graph:
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/f7d4b48b-2dd6-4696-84a7-8fe6a4230f31)
## KMeans clusters:
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/f1ab98f0-3d6d-44d3-a2d1-8d2dc359f320)
## y_pred:
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/c8dcdbe2-4ca3-4549-ae68-b91bdc52b4f8)
## Customers Segments Graph:
![image](https://github.com/Mythilidharman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119104110/d8e6e1f4-e7e8-4b46-a6a2-7bbe09f427f8)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
