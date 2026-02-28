import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self,K=3):
        self.k = K
        self.centroids = None

    @staticmethod
    def euclidian_distance(data_point,centroid):
        return np.sqrt(np.sum( (centroid - data_point)**2 ,axis =1))

    def fit(self,X,Max_Itr):

        self.centroids=np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),size=(self.k,X.shape[1]))

        for _ in range(Max_Itr):
            y =[]           #y[i] = cluster number assigned to X[i]
            for single_data_point in X:
                distances = KMeans.euclidian_distance(single_data_point,self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            cluster_indices = []            # indices of points that belong to that cluster
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y==i))

            cluster_centers =[]

            for i , indices in enumerate(cluster_indices):
                if len(indices) ==0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices],axis=0)[0])
                
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return  y




X= np.random.randint(0,100,(100,2))

kmeans = KMeans(K=3)

lables = kmeans.fit(X, Max_Itr=1000)

plt.scatter(X[:,0],X[:,1],c=lables)
plt.scatter(kmeans.centroids[:,0],kmeans.centroids[:,1],c=range((len(kmeans.centroids))),marker="*",s=200)

plt.show()
