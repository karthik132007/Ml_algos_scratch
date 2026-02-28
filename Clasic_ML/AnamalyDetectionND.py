import numpy as np

class GaussianAnomalyDetector:
    def fit(self,X):
        self.mu = np.mean(X,axis=0)
        self.cov = np.cov(X,rowvar=False)  # covariance matrix
        self.cov += 1e-6*np.eye(self.cov.shape[0])  #add small values in diagonal of matrix so det won't be 0
        self.inv_cov = np.linalg.inv(self.cov)
        self.d= X.shape[1]  # no of features
        self.det_cov = np.linalg.det(self.cov)  # det for cov matrix

    def pdf(self,x):
        x_mu = x- self.mu
        const = 1 / np.sqrt((2 * np.pi) ** self.d * self.det_cov)
        exp = np.exp(
            -0.5 * np.sum((x_mu @ self.inv_cov) * x_mu,axis=1)
        )
        return const*exp
    def predict(self,x,e):
        probs = self.pdf(x)
        return probs<e

X_train = np.array([
    [18, 35, 120],
    [20, 38, 125],
    [22, 40, 130],
    [19, 36, 118],
    [21, 39, 128],
    [23, 42, 135],
    [20, 37, 122],
    [22, 41, 132],
    [19, 34, 115],
    [21, 40, 129]
])

model = GaussianAnomalyDetector()

model .fit(X_train)

X_test = np.array([
    [20, 38, 126],   # normal
    [22, 41, 133],   # normal
    [60, 85, 400],   # anomaly (high everything)
    [10, 90, 50],    # anomaly (weird combo)
    [19, 36, 119]    # normal
])
print("before normalization")
print(model.predict(X_test,0.01))

print("after normalization")

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

model.fit(X_train)
print(model.predict(X_test,0.01))
