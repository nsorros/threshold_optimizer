from scipy.sparse import csr_matrix, save_npz
import numpy as np
import typer


def create_data(rows: int, columns: int, y_pred_path="Y_pred.npz", y_test_path="Y_test.npz"):
    Y_pred_proba = np.random.randn(rows, columns)

    # sparsifying probs
    Y_pred_proba[Y_pred_proba < 0.2] = 0
    Y_pred_proba = csr_matrix(Y_pred_proba)

    thresholds = 0.5 #np.random.randn(columns)
    Y_test = csr_matrix(Y_pred_proba > thresholds)

    save_npz(y_pred_path, Y_pred_proba)
    save_npz(y_test_path, Y_test)

if __name__ == "__main__":
    typer.run(create_data)
