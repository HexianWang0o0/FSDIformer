import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true, return_mean=True):
    _logits = np.abs(pred - true)
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSE(pred, true, return_mean=True):
    _logits = (pred - true) ** 2
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def RMSE(pred, true, return_mean=True):
    return np.sqrt(MSE(pred, true, return_mean=return_mean))


def MAPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.abs((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.square((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits

def MAE_2(pred, true, return_mean=True):
    pred = pred.squeeze(-1)
    true = true.squeeze(-1)

    valid_mask = true.max(dim=1).values > 0
    valid_predictions = pred[valid_mask]
    valid_true_values = true[valid_mask]

    absolute_diff = np.abs(valid_predictions - valid_true_values)

    max_true_values = np.max(valid_true_values, axis=1)
    normalized_error = (np.sum(absolute_diff, axis=1) / max_true_values) / pred.shape[1]
    average_error = np.mean(normalized_error)

    return average_error

def WRMSE(pred, true, return_mean=True):
    pred = pred.squeeze(-1)
    true = true.squeeze(-1)

    valid_mask = true.max(dim=1).values > 0
    valid_predictions = pred[valid_mask]
    valid_true_values = true[valid_mask]

    absolute_diff = np.abs(valid_predictions - valid_true_values)  # 每个时间步的绝对误差
    squared_diff = absolute_diff ** 2  # 差的平方

    total_absolute_diff = np.sum(absolute_diff, axis=1)
    weights = absolute_diff / total_absolute_diff[:, np.newaxis]

    weighted_squared_diff = weights * squared_diff
    final_error = np.sqrt(np.sum(weighted_squared_diff, axis=1))

    max_true_values = np.max(valid_true_values, axis=1)
    normalized_final_error = final_error / max_true_values

    average_error = np.mean(normalized_final_error)

    return average_error

def PHA(pred, true, return_mean=True):
    mae_2 = MAE_2(pred, true)
    wrmse = WRMSE(pred, true)
    pha = (0.4 * (1 - wrmse) + 0.6 * (1 - mae_2)) * 100
    return pha


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mae_2 = MAE_2(pred, true)
    wrmse = WRMSE(pred, true)
    pha = (0.4 * (1 - wrmse) + 0.6 * (1 - mae_2)) * 100

    return mae, mse, mae_2, wrmse, pha
