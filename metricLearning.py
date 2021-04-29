from metric_learn import LMNN, NCA, LFDA, ITML, SDML, RCA, LSML, MMC, MLKR


methods = {'lmnn': LMNN,
           'nca': NCA,
           'lfda': LFDA,
           'itml': ITML,
           'sdml': SDML,
           'rca': RCA,
           'lsml': LSML,
           'mmc': MMC,
           'mlkr': MLKR}


def run(X, y, method='lmnn', **kwargs):
    try:
        obj = methods[method](**kwargs)
    except KeyError:
        raise ValueError(f'Method {method} not implemented!')
    obj.fit(X, y)
    return obj.get_metric()
