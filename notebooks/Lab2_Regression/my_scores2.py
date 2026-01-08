import numpy as np

def my_r2_score(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

def my_MSE(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean((y - y_pred) ** 2)

def my_RMSE(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y - y_pred) ** 2))

def my_MAE(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y-y_pred))

def my_MAPE(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / y))

def calc_my_scores(y, y_pred):
    scores = [
        my_MAE(y, y_pred),
        my_MSE(y, y_pred),
        my_RMSE(y, y_pred),
        my_MAPE(y, y_pred),
        my_r2_score(y, y_pred)
    ]
    scores = map(lambda x: round(x, 2), scores)
    return scores

def calc_test_train_my_scores(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_scores = calc_my_scores(y_test, y_test_pred)
    return test_scores
