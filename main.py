from KNN import runKNN
from dataset import load_data


X, X_t, y, y_t = load_data()
score = runKNN(X, y, X_t, y_t, 5, 'euclidean')
print(score)
