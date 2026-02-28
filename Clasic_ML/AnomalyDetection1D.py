import numpy as np


class GaussianAnamalyDetector:
    def fit(self,X):
        self.mu = np.mean(X)
        self.var = np.var(X)

    def pdf(self,X):
        const = 1/np.sqrt(2*np.pi*self.var)

        coff = np.exp(-((X-self.mu))**2/(2*self.var))
        return const*coff
    def predict(self,x,e):

        probs = self.pdf(x)

        return probs<e


model =  GaussianAnamalyDetector()

X_train = np.array([10, 11, 9, 10, 12, 11, 10])

model.fit(X_train)

X_test = np.array([10, 11, 50])
print(model.predict(X_test,0.01))
