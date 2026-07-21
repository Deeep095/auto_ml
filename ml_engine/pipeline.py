from sklearn.pipeline import Pipeline

from auto_ml.ml_engine.preprocessing import preprocess_request

from auto_ml.ml_engine.metrics import (
    evaluate_model,
    metric_to_score
)

from auto_ml.ml_engine.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    CLUSTERING_MODELS,
    FORECASTING_MODELS
)

from auto_ml.ml_engine.plots import (
    PLOT_REGISTRY
)

from auto_ml.ml_engine.reports import (
    generate_report
)

from auto_ml.ml_engine.utils import (
    create_directories,
    create_experiment_directory,
    save_model
)


def get_model_registry(problem_type):

    if problem_type == "classification":

        return CLASSIFICATION_MODELS

    if problem_type == "regression":

        return REGRESSION_MODELS

    if problem_type == "forecasting":

        return FORECASTING_MODELS

    if problem_type == "clustering":

        return CLUSTERING_MODELS

    raise Exception(

        f"Unsupported problem type : {problem_type}"

    )


def train_models(

    execution_plan,

    prepared_data

):

    registry = get_model_registry(

        execution_plan["problem_type"]

    )

    X_train = prepared_data["X_train"]

    X_test = prepared_data["X_test"]

    y_train = prepared_data["y_train"]

    y_test = prepared_data["y_test"]

    preprocessor = prepared_data["preprocessor"]

    trained_models = {}

    predictions = {}

    results = []

    best_pipeline = None

    best_score = -999999

    best_model_name = None

    for model_info in execution_plan["recommended_models"]:

        model_name = model_info["name"]

        if model_name not in registry:

            continue

        estimator = registry[model_name]

        pipeline = Pipeline(

            [

                (

                    "preprocessor",

                    preprocessor

                ),

                (

                    "classifier",

                    estimator

                )

            ]

        )

        pipeline.fit(

            X_train,

            y_train

        )

        y_pred = pipeline.predict(

            X_test

        )

        metrics = evaluate_model(

            execution_plan["problem_type"],

            y_test,

            y_pred

        )

        score = metric_to_score(

            metrics,

            execution_plan["problem_type"]

        )

        result = {

            "model": model_name,

            **metrics

        }

        trained_models[model_name] = pipeline

        predictions[model_name] = y_pred

        results.append(

            result

        )

        if score > best_score:

            best_score = score

            best_pipeline = pipeline

            best_model_name = model_name

    results.sort(

        key=lambda x: metric_to_score(

            x,

            execution_plan["problem_type"]

        ),

        reverse=True

    )

    return {

        "results": results,

        "trained_models": trained_models,

        "predictions": predictions,

        "best_model": best_pipeline,

        "best_model_name": best_model_name

    }
def generate_visualizations(

    execution_plan,

    training_output,

    prepared_data,

    experiment_dir

):

    artifacts = {}

    visualizations = execution_plan.get(

        "visualizations",

        []

    )

    best_model_name = training_output[

        "best_model_name"

    ]

    best_pipeline = training_output[

        "trained_models"

    ][

        best_model_name

    ]

    y_pred = training_output[

        "predictions"

    ][

        best_model_name

    ]

    for visualization in visualizations:

        visualization_type = visualization[

            "type"

        ]

        if visualization_type not in PLOT_REGISTRY:

            continue

        output = PLOT_REGISTRY[

            visualization_type

        ](

            artifact_dir=experiment_dir,

            model=best_pipeline,

            model_name=best_model_name,

            results=training_output["results"],

            X_test=prepared_data["X_test"],

            y_test=prepared_data["y_test"],

            y_pred=y_pred,

            options=visualization

        )

        if output is not None:

            artifacts[

                visualization_type

            ] = str(output)

    return artifacts


def run_pipeline(

    dataset_path,

    execution_plan

):

    create_directories()

    experiment_dir = create_experiment_directory()

    prepared_data = preprocess_request(

        dataset_path,

        execution_plan

    )

    training_output = train_models(

        execution_plan,

        prepared_data

    )

    best_pipeline = training_output[

        "best_model"

    ]

    save_model(

        best_pipeline,

        experiment_dir

    )

    artifacts = generate_visualizations(

        execution_plan,

        training_output,

        prepared_data,

        experiment_dir

    )

    report_files = generate_report(

        dataset_shape=list(

            prepared_data["dataframe"].shape

        ),

        target_column=execution_plan[

            "target_column"

        ],

        problem_type=execution_plan[

            "problem_type"

        ],

        best_model=training_output[

            "results"

        ][0],

        all_models=training_output[

            "results"

        ],

        artifacts=artifacts

    )

    artifacts.update(

        report_files

    )

    response = {

        "dataset_shape": list(

            prepared_data["dataframe"].shape

        ),

        "problem_type": execution_plan[

            "problem_type"

        ],

        "target_column": execution_plan[

            "target_column"

        ],

        "best_model": training_output["results"][0],

        "all_models": training_output["results"],

        "artifacts": artifacts

    }

    return response