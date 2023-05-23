
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import multiprocessing
import logging

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    parameters = {
        'n_estimators': [20, 25, 30],
        'max_depth': [3, 6],
        'min_samples_split': [20, 50, 100],
        'learning_rate': [1.0],
    }

    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching hyperparameters on {} cores".format(njobs))

    model = GridSearchCV(GradientBoostingClassifier(random_state=0),
                       param_grid=parameters,
                       cv=3,
                       n_jobs=njobs,
                       verbose=2,
                       )
    
    model.fit(X_train, y_train)
    logging.info("Best parameters found")
    logging.info("best parameters: {}".format(model.best_params_))

    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions

def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and real values.
    ------
    y : np.array
    preds : np.array
    Returns
    ------
    cm : confusion matrix.
    """
    cm = confusion_matrix(y, preds)
    return cm

def slice_metrics(df, feat, y, preds):
    """
    Compute precision, recall, and F1 score for each unique value of a categorical feature in a dataframe.
    Args:
        df: input dataframe with categorical feature column
        feat: name of the categorical feature column
        y: array of true labels
        preds: array of predicted labels
    Returns:
        Dataframe with columns:
            - value: unique values of the categorical feature
            - samples: number of data samples in the slice
            - prec: precision score
            - rec: recall score
            - f1: F1 score
    """
    vals = df[feat].unique()
    data = []
    for v in vals:
        mask = df[feat] == v
        slice_y, slice_preds = y[mask], preds[mask]
        prec, rec, f1 = compute_model_metrics(slice_y, slice_preds)
        data.append((v, len(slice_y), prec, rec, f1))
    
    return pd.DataFrame(data, columns=['value', 'samples', 'prec', 'rec', 'f1']).set_index('value')

