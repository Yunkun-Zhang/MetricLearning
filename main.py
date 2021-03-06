from KNN import runKNN, reduce_by_metric
from dataPreprocessing import load_data
from metricLearning import run
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='Metric learning method', default='nca')
parser.add_argument('-k', '--k', help='Number of neighbors to consider', default=3)
parser.add_argument('-d', '--dim', help='Dimensionality of lda space', default=None)
parser.add_argument('-r', '--reduce', help='Whether to use dim-reduced data', action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    method = args.method
    k = args.k
    dim = int(args.dim) if args.dim else None

    # decide which arguments to add
    kwargs = dict()
    if method in ['lmnn']:
        kwargs['k'] = k
    if method in ['lmnn', 'nca', 'lfda']:
        kwargs['n_components'] = dim
    if method == 'sdml':
        kwargs['balance_param'] = 1e-5
        kwargs['sparsity_param'] = 1e-5

    # load data
    X, X_t, y, y_t = load_data(reduce=args.reduce)

    print('Performing metric learning...', end='\r')
    s = time.time()
    metric = run(X, y, method=method, **kwargs)
    e = time.time() - s
    print(f'Performing metric learning...Done. Time: {e:.4f}s.')

    X = reduce_by_metric(X, metric)
    X_t = reduce_by_metric(X_t, metric)

    print('Running KNN...', end='\r')
    score = runKNN(X, y, X_t, y_t, k=k)
    print(f'Running KNN...Done. KNN score: {score:.6f}.')

    with open('results.txt', 'a') as f:
        f.write(f'score: {score:.6f}, time: {e:.4f} (method={method}, k={k}, reduce={args.reduce})\n')
