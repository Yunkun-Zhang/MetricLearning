from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from multiprocessing import Pool
from dataset import load_data


def runKNN(X=None, y=None, X_t=None, y_t=None, k=5, metric='minkowski'):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X, y)
    score = knn.score(X_t, y_t)
    return score


def runKFold(X, y, k, metric):
    kf = KFold(n_splits=5)
    s = 0
    for (X_train, X_test), (y_train, y_test) in zip(kf.split(X), kf.split(y)):
        s += runKNN(X[X_train], y[y_train], X[X_test], y[y_test], k, metric)
    return s / 5


def get_best_k(X, y, metric):
    k_range = [3, 5, 7]
    result = []
    ps = Pool(len(k_range))
    for k in k_range:
        result.append((k, ps.apply_async(runKFold, args=(X, y, k, metric)).get()))
    ps.close()
    ps.join()
    return result


if __name__ == '__main__':
    X, X_t, y, y_t = load_data()
    print(get_best_k(X, y, 'euclidean'))
