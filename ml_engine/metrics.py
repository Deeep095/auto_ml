import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

    mean_absolute_error,

    mean_squared_error,

    r2_score,

    silhouette_score,

    davies_bouldin_score,

    calinski_harabasz_score

)


def classification_metrics(

    y_true,

    y_pred

):

    return {

        "accuracy": round(

            float(

                accuracy_score(

                    y_true,

                    y_pred

                )

            ),

            4

        ),

        "precision": round(

            float(

                precision_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                )

            ),

            4

        ),

        "recall": round(

            float(

                recall_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                )

            ),

            4

        ),

        "f1": round(

            float(

                f1_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                )

            ),

            4

        )

    }


def regression_metrics(

    y_true,

    y_pred

):

    mae = mean_absolute_error(

        y_true,

        y_pred

    )

    mse = mean_squared_error(

        y_true,

        y_pred

    )

    rmse = np.sqrt(

        mse

    )

    r2 = r2_score(

        y_true,

        y_pred

    )

    return {

        "mae": round(

            float(mae),

            4

        ),

        "mse": round(

            float(mse),

            4

        ),

        "rmse": round(

            float(rmse),

            4

        ),

        "r2": round(

            float(r2),

            4

        )

    }


def forecasting_metrics(

    y_true,

    y_pred

):

    mae = mean_absolute_error(

        y_true,

        y_pred

    )

    mse = mean_squared_error(

        y_true,

        y_pred

    )

    rmse = np.sqrt(

        mse

    )

    mask = y_true != 0

    if np.sum(mask) == 0:

        mape = 0

    else:

        mape = np.mean(

            np.abs(

                (

                    y_true[mask] -

                    y_pred[mask]

                )

                /

                y_true[mask]

            )

        ) * 100

    return {

        "mae": round(

            float(mae),

            4

        ),

        "rmse": round(

            float(rmse),

            4

        ),

        "mape": round(

            float(mape),

            4

        )

    }


def clustering_metrics(

    X,

    labels

):

    unique_labels = np.unique(

        labels

    )

    if len(unique_labels) <= 1:

        return {

            "silhouette_score": None,

            "davies_bouldin_score": None,

            "calinski_harabasz_score": None

        }

    silhouette = silhouette_score(

        X,

        labels

    )

    davies = davies_bouldin_score(

        X,

        labels

    )

    calinski = calinski_harabasz_score(

        X,

        labels

    )

    return {

        "silhouette_score": round(

            float(silhouette),

            4

        ),

        "davies_bouldin_score": round(

            float(davies),

            4

        ),

        "calinski_harabasz_score": round(

            float(calinski),

            4

        )

    }


def get_confusion_matrix(

    y_true,

    y_pred

):

    return confusion_matrix(

        y_true,

        y_pred

    ).tolist()


def get_classification_report(

    y_true,

    y_pred

):

    return classification_report(

        y_true,

        y_pred,

        output_dict=True,

        zero_division=0

    )


def evaluate_model(

    problem_type,

    y_true,

    y_pred,

    X=None,

    labels=None

):

    if problem_type == "classification":

        return classification_metrics(

            y_true,

            y_pred

        )

    if problem_type == "regression":

        return regression_metrics(

            y_true,

            y_pred

        )

    if problem_type == "forecasting":

        return forecasting_metrics(

            y_true,

            y_pred

        )

    if problem_type == "clustering":

        return clustering_metrics(

            X,

            labels

        )

    raise ValueError(

        f"Unsupported problem type: {problem_type}"

    )


def metric_to_score(

    metrics,

    problem_type

):

    if problem_type == "classification":

        return metrics["accuracy"]

    if problem_type == "regression":

        return -metrics["rmse"]

    if problem_type == "forecasting":

        return -metrics["rmse"]

    if problem_type == "clustering":

        score = metrics.get(

            "silhouette_score"

        )

        return -999 if score is None else score

    return -999