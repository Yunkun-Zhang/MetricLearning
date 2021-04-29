from KNN import runKNN
from dataPreprocessing import load_data
from metricLearning import run
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='Metric learning method', default='nca')
parser.add_argument('-k', '--k', help='Number of neighbors to consider', default=3)
parser.add_argument('-d', '--dim', help='Dimensionality of reduced space', default=None)
args = parser.parse_args()

if __name__ == '__main__':
    method = args.method
    k = args.k
    dim = int(args.dim) if args.dim else None

    # decide which arguments to add
    kwargs = dict()
    reduce = False
    if method in []:
        kwargs['k'] = k
    if method in []:
        kwargs['n_components'] = dim
    if method in []:
        reduce = True

    # load data
    X, X_t, y, y_t = load_data(reduce=reduce)

    print('Performing metric learning...', end='\r')
    s = time.time()
    metric = run(X, y, method=method, **kwargs)
    e = time.time() - s
    print(f'Performing metric learning...Done. Time: {e}s.')

    print('Running KNN...', end='\r')
    score = runKNN(X, y, X_t, y_t, k=k, metric=metric)
    print(f'Running KNN...Done. KNN score: {score:.6f}.')

    with open('results.txt', 'a') as f:
        f.write(f'score: {score:.6f} (method={method}, k={k}, dim={dim})\n')
