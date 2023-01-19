import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("project118.csv")

graph = px.scatter(df , x = "Size" , y ="Light")

#graph.show()

x = df.iloc[: ,[0,1]].values

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters= i , init="k-means++")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

kmeans = KMeans(n_clusters= 3 , init="k-means++")
y_kmeans = kmeans.fit_predict(x)

plt.figure(figsize=(15,7))
sns.scatterplot(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.title('Clusters of Interstellar Objects')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show()



