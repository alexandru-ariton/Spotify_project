import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                t[v].fillna(t[v].mode()[0], inplace=True)


def unique(a):
    k = np.unique(a, return_index=True)[1]
    return [a[i] for i in sorted(k)]


def partitie(h, k=None):
    m = np.shape(h)[0]
    n = m + 1
    if k is None:
        d = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(d)
        k = m - j
    else:
        j = m - k
    threshold = (h[j, 2] + h[j + 1, 2]) / 2
    c = np.arange(n)
    for i in range(m - k + 1):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k2] = n + i
        c[c == k1] = n + i
    coduri = pd.Categorical(c).codes
    p = np.array(["C" + str(i + 1) for i in coduri])
    return k, threshold, p
