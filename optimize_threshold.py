from functools import partial
import time

from scipy.sparse import load_npz
import numpy as np
import typer

if "line_profiler" not in dir() and "profile" not in dir():
    # no-op profile decorator
    def profile(f):
        return f

@profile
def confusion_matrix(y_test, y_pred):
    tp = y_test.dot(y_pred)
    fp = y_pred.sum() - tp
    fn = y_test.sum() - tp
    tn = y_test.shape[0] - tp - fp - fn
    return np.array([[tn, fp], [fn, tp]])

@profile
def multilabel_confusion_matrix(Y_test, Y_pred):
    tp = Y_test.multiply(Y_pred).sum(axis=0)
    fp = Y_pred.sum(axis=0) - tp
    fn = Y_test.sum(axis=0) - tp
    tn = Y_test.shape[0] - tp - fp - fn
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

@profile
def f(y_pred_proba, y_test, mlcm, k, thresholds):
    threshold = thresholds[k]

    y_pred = y_pred_proba > threshold
    cmk = confusion_matrix(y_test, y_pred)
    mlcm[k,:,:] = cmk
 
    cm = mlcm.sum(axis=0)
    tn, fp, fn, tp = cm.ravel()
    f1 = tp / ( tp+ (fp+fn) / 2)
    return f1

@profile
def argmaxf1(y_pred_proba, y_test, optimal_thresholds, mlcm, k, nb_thresholds=None):
    optimal_thresholds_star = optimal_thresholds.copy()

    fp = partial(f, y_pred_proba, y_test, mlcm, k)
    
    if nb_thresholds:
        thresholds = np.array([i/nb_thresholds for i in range(0, nb_thresholds)])
    else:
        thresholds = np.unique(y_pred_proba)
    for threshold in thresholds:
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

    Y_pred = Y_pred_proba > optimal_thresholds
    mlcm = multilabel_confusion_matrix(Y_test, Y_pred)

    updated = True    
    while updated:
        updated = False
        for k in range(N):
            start = time.time()

            y_pred_proba = np.array(Y_pred_proba[:,k].todense()).ravel()
            y_test = np.array(Y_test[:,k].todense()).ravel()
            fp = partial(f, y_pred_proba, y_test, mlcm, k)

            optimal_thresholds_star = argmaxf1(y_pred_proba, y_test, optimal_thresholds, mlcm, k, nb_thresholds)

            if fp(optimal_thresholds_star) > fp(optimal_thresholds):
                optimal_thresholds = optimal_thresholds_star
                Y_pred = Y_pred_proba > optimal_thresholds
                mlcm = multilabel_confusion_matrix(Y_test, Y_pred)
                updated = True

            print(f"label {k} - updated {updated} - time elapsed {time.time()-start:.2f}s")

if __name__ == "__main__":
    typer.run(optimize_threshold)
