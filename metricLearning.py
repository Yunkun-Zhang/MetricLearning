from metric_learn import LMNN, NCA, LFDA


def run(X, y, method='lmnn', k=3, d=None):
    if method == 'lmnn':
        obj = LMNN(k=k)
    elif method == 'nca':
        obj = NCA(n_components=d)
    elif method == 'lfda':
        obj = LFDA(n_components=d, k=k)
    else:
        raise ValueError(f'Method {method} not implemented!')
    obj.fit(X, y)
    return obj.get_metric()
