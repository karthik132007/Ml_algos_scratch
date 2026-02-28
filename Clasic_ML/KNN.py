import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def eu_distance(p,q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))

class KNN:
    def __init__(self,k):
        self.k=k
        self.points=None

    def fit(self,points):
        self.points=points

    def predict(self,new_point):
        distance =[]
        for catogery in self.points:
            for point in self.points[catogery]:
                dist=eu_distance(new_point,point)
                distance.append([dist,catogery])
        catogories=[catogery[1] for catogery in sorted(distance)[:self.k]]
        result=Counter(catogories).most_common(1)[0][0]
        return result
clf= KNN(k=3)

points = {
    "red":[[2,4],[1,3],[2,3],[3,2],[2,1],[1,3],[2,2]],
    "blue":[[5,6],[4,5],[4,6],[6,6],[5,4],[3,4],[4,3]]
}
new_point=[3,3]

clf.fit(points)
print(clf.predict(new_point))

for category in points:
    for point in points[category]:
        plt.scatter(point[0], point[1], color=category)
plt.scatter(new_point[0], new_point[1], color="green", marker="*", s=200)
plt.title(f"KNN Prediction → {clf.predict(new_point)}")

plt.grid(True)
plt.show()
