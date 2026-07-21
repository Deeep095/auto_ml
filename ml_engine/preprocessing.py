import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.impute import (
    SimpleImputer
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OrdinalEncoder
)

from sklearn.model_selection import (
    train_test_split
)

from auto_ml.ml_engine.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE
)


def load_dataset(dataset_path):

    if dataset_path.endswith(".csv"):

        return pd.read_csv(dataset_path)

    if dataset_path.endswith(".xlsx"):

        return pd.read_excel(dataset_path)

    raise Exception(
        "Unsupported dataset format."
    )


def split_features_target(

    dataframe,

    target_column

):

    X = dataframe.drop(
        columns=[target_column]
    )

    y = dataframe[target_column]

    return X, y


def detect_column_types(X):

    numerical_columns = X.select_dtypes(

        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]

    ).columns.tolist()

    categorical_columns = X.select_dtypes(

        include=[
            "object",
            "category",
            "bool"
        ]

    ).columns.tolist()

    return (

        numerical_columns,

        categorical_columns

    )


def build_numeric_pipeline(

    preprocessing_plan

):

    steps = []

    missing_strategy = preprocessing_plan.get(

        "missing_values",

        "mean"

    )

    scaler = preprocessing_plan.get(

        "scaling",

        "standard"

    )

    steps.append(

        (

            "imputer",

            SimpleImputer(

                strategy=missing_strategy

            )

        )

    )

    if scaler == "standard":

        steps.append(

            (

                "scaler",

                StandardScaler()

            )

        )

    elif scaler == "minmax":

        steps.append(

            (

                "scaler",

                MinMaxScaler()

            )

        )

    elif scaler == "robust":

        steps.append(

            (

                "scaler",

                RobustScaler()

            )

        )

    return Pipeline(

        steps

    )


def build_categorical_pipeline(

    preprocessing_plan

):

    steps = []

    missing_strategy = preprocessing_plan.get(

        "categorical_missing",

        "most_frequent"

    )

    encoder = preprocessing_plan.get(

        "encoding",

        "onehot"

    )

    steps.append(

        (

            "imputer",

            SimpleImputer(

                strategy=missing_strategy

            )

        )

    )

    if encoder == "onehot":

        steps.append(

            (

                "encoder",

                OneHotEncoder(

                    handle_unknown="ignore",
                    sparse_output=False
                )

            )

        )

    elif encoder == "ordinal":

        steps.append(

            (

                "encoder",

                OrdinalEncoder()

            )

        )

    return Pipeline(

        steps

    )


def build_preprocessor(

    X,

    execution_plan

):

    (

        numerical_columns,

        categorical_columns

    ) = detect_column_types(

        X

    )

    preprocessing_plan = execution_plan.get(

        "preprocessing",

        {}

    )

    numeric_pipeline = build_numeric_pipeline(

        preprocessing_plan

    )

    categorical_pipeline = build_categorical_pipeline(

        preprocessing_plan

    )

    transformer = ColumnTransformer(

        transformers=[

            (

                "num",

                numeric_pipeline,

                numerical_columns

            ),

            (

                "cat",

                categorical_pipeline,

                categorical_columns

            )

        ]

    )

    return transformer


def split_dataset(

    X,

    y,

    execution_plan

):

    test_size = execution_plan.get(

        "test_size",

        DEFAULT_TEST_SIZE

    )

    random_state = execution_plan.get(

        "random_state",

        DEFAULT_RANDOM_STATE

    )

    problem_type = execution_plan.get(

        "problem_type",

        "classification"

    )

    if problem_type == "classification":

        return train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y

        )

    return train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state

    )


def prepare_dataset(

    dataset_path,

    execution_plan

):

    dataframe = load_dataset(

        dataset_path

    )

    target_column = execution_plan[

        "target_column"

    ]

    X, y = split_features_target(

        dataframe,

        target_column

    )

    preprocessor = build_preprocessor(

        X,

        execution_plan

    )

    (

        X_train,

        X_test,

        y_train,

        y_test

    ) = split_dataset(

        X,

        y,

        execution_plan

    )

    return {

        "dataframe": dataframe,

        "X": X,

        "y": y,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "preprocessor": preprocessor

    }


def dataset_summary(

    dataframe,

    target_column

):

    numerical_columns = dataframe.select_dtypes(

        include=[

            "number"

        ]

    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(

        exclude=[

            "number"

        ]

    ).columns.tolist()

    summary = {

        "rows": len(dataframe),

        "columns": len(dataframe.columns),

        "target_column": target_column,

        "numerical_columns": numerical_columns,

        "categorical_columns": categorical_columns,

        "missing_values": dataframe.isnull().sum().to_dict(),

        "dtypes": dataframe.dtypes.astype(str).to_dict()

    }

    return summary


def class_distribution(

    dataframe,

    target_column

):

    if target_column not in dataframe.columns:

        return {}

    counts = dataframe[

        target_column

    ].value_counts()

    return counts.to_dict()


def missing_value_summary(

    dataframe

):

    return dataframe.isnull().sum().to_dict()


def feature_summary(

    dataframe

):

    summary = {}

    for column in dataframe.columns:

        summary[column] = {

            "dtype": str(

                dataframe[column].dtype

            ),

            "missing": int(

                dataframe[column].isnull().sum()

            ),

            "unique": int(

                dataframe[column].nunique()

            )

        }

    return summary


def preprocess_request(

    dataset_path,

    execution_plan

):

    prepared = prepare_dataset(

        dataset_path,

        execution_plan

    )

    dataframe = prepared[

        "dataframe"

    ]

    target_column = execution_plan[

        "target_column"

    ]

    prepared["summary"] = dataset_summary(

        dataframe,

        target_column

    )

    prepared["class_distribution"] = class_distribution(

        dataframe,

        target_column

    )

    prepared["feature_summary"] = feature_summary(

        dataframe

    )

    prepared["missing_summary"] = missing_value_summary(

        dataframe

    )

    return prepared