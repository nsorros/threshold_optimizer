from functools import partial
import time

from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import numpy as np
import typer

if "line_profiler" not in dir() and "profile" not in dir():
    # no-op profile decorator
    def profile(f):
        return f

@profile
def f(Y_pred_proba, Y_test, thresholds):
    Y_pred = Y_pred_proba > thresholds
    return f1_score(Y_test, Y_pred, average="micro")

@profile
def argmaxf1(Y_pred_proba, Y_test, optimal_thresholds, k, nb_thresholds=None):
    optimal_thresholds_star = optimal_thresholds.copy()

    fp = partial(f, Y_pred_proba, Y_test)
    
    if nb_thresholds:
        thresholds = np.array([i/nb_thresholds for i in range(0, nb_thresholds)])
    else:
        thresholds = np.unique(np.array(Y_pred_proba[:,k].todense()).ravel())
    for threshold in thresholds:
        print(threshold)
        optimal_thresholds_star[k] = threshold

        if fp(optimal_thresholds_star) > fp(optimal_thresholds):
            optimal_thresholds = optimal_thresholds_star.copy()

    return optimal_thresholds

@profile
def optimize_threshold(y_pred_path, y_test_path, nb_thresholds:int=None):
    Y_pred_proba = load_npz(y_pred_path)
    Y_test = load_npz(y_test_path)

    N = Y_pred_proba.shape[1]

    optimal_thresholds = np.array(Y_pred_proba.min(axis=1).todense())

    fp = partial(f, Y_pred_proba, Y_test)

    updated = True    
    while updated:
        updated = False
        for k in range(N):
            start = time.time()
            optimal_thresholds_star = argmaxf1(Y_pred_proba, Y_test, optimal_thresholds, k, nb_thresholds)

            if fp(optimal_thresholds_star) > fp(optimal_thresholds):
                optimal_thresholds = optimal_thresholds_star
                updated = True
            print(f"label {k} - updated {updated} - time elapsed {time.time()-start:.2f}s")

if __name__ == "__main__":
    typer.run(optimize_threshold)
