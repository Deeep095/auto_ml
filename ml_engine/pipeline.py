from sklearn.pipeline import Pipeline

from preprocessing import preprocess_request

from metrics import (
    evaluate_model,
    metric_to_score
)

from models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    CLUSTERING_MODELS,
    FORECASTING_MODELS
)

from plots import (
    PLOT_REGISTRY
)

from reports import (
    generate_report
)

from utils import (
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


import time


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

    total_start = time.time()

    for model_info in execution_plan["recommended_models"]:

        model_name = model_info["name"]

        if model_name not in registry:

            continue

        print("=" * 60, flush=True)
        print(f"Training {model_name}...", flush=True)

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

        start = time.time()

        pipeline.fit(

            X_train,

            y_train

        )

        fit_time = time.time() - start

        print(
            f"✓ Training completed in {fit_time:.2f} seconds",
            flush=True
        )

        start = time.time()

        y_pred = pipeline.predict(

            X_test

        )

        predict_time = time.time() - start

        print(
            f"✓ Prediction completed in {predict_time:.2f} seconds",
            flush=True
        )

        metrics = evaluate_model(

            execution_plan["problem_type"],

            y_test,

            y_pred

        )

        print(
            f"✓ Metrics: {metrics}",
            flush=True
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

        print(
            f"Finished {model_name} "
            f"(Total: {fit_time + predict_time:.2f}s)",
            flush=True
        )

    print("=" * 60, flush=True)
    print(
        f"All models completed in {time.time() - total_start:.2f} seconds",
        flush=True
    )

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

    total_start = time.time()

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

        print("=" * 60, flush=True)
        print(
            f"Generating {visualization_type}...",
            flush=True
        )

        start = time.time()

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

        elapsed = time.time() - start

        print(
            f"✓ {visualization_type} generated in {elapsed:.2f} seconds",
            flush=True
        )

        if output is not None:

            artifacts[

                visualization_type

            ] = str(output)

    print("=" * 60, flush=True)
    print(
        f"All visualizations completed in {time.time() - total_start:.2f} seconds",
        flush=True
    )

    return artifacts


# def generate_visualizations(

#     execution_plan,

#     training_output,

#     prepared_data,

#     experiment_dir

# ):

#     artifacts = {}

#     visualizations = execution_plan.get(

#         "visualizations",

#         []

#     )

#     best_model_name = training_output[

#         "best_model_name"

#     ]

#     best_pipeline = training_output[

#         "trained_models"

#     ][

#         best_model_name

#     ]

#     y_pred = training_output[

#         "predictions"

#     ][

#         best_model_name

#     ]

#     for visualization in visualizations:

#         visualization_type = visualization[

#             "type"

#         ]

#         if visualization_type not in PLOT_REGISTRY:

#             continue

#         output = PLOT_REGISTRY[

#             visualization_type

#         ](

#             artifact_dir=experiment_dir,

#             model=best_pipeline,

#             model_name=best_model_name,

#             results=training_output["results"],

#             X_test=prepared_data["X_test"],

#             y_test=prepared_data["y_test"],

#             y_pred=y_pred,

#             options=visualization

#         )

#         if output is not None:

#             artifacts[

#                 visualization_type

#             ] = str(output)

#     return artifacts


# def run_pipeline(

#     dataset_path,

#     execution_plan

# ):

#     create_directories()

#     experiment_dir = create_experiment_directory()

#     prepared_data = preprocess_request(

#         dataset_path,

#         execution_plan

#     )

#     training_output = train_models(

#         execution_plan,

#         prepared_data

#     )

#     best_pipeline = training_output[

#         "best_model"

#     ]

#     save_model(

#         best_pipeline,

#         experiment_dir

#     )

#     artifacts = generate_visualizations(

#         execution_plan,

#         training_output,

#         prepared_data,

#         experiment_dir

#     )

#     report_files = generate_report(

#         dataset_shape=list(

#             prepared_data["dataframe"].shape

#         ),

#         target_column=execution_plan[

#             "target_column"

#         ],

#         problem_type=execution_plan[

#             "problem_type"

#         ],

#         best_model=training_output[

#             "results"

#         ][0],

#         all_models=training_output[

#             "results"

#         ],

#         artifacts=artifacts

#     )

#     artifacts.update(

#         report_files

#     )

#     response = {

#         "dataset_shape": list(

#             prepared_data["dataframe"].shape

#         ),

#         "problem_type": execution_plan[

#             "problem_type"

#         ],

#         "target_column": execution_plan[

#             "target_column"

#         ],

#         "best_model": training_output["results"][0],

#         "all_models": training_output["results"],

#         "artifacts": artifacts

#     }

#     return response









def run_pipeline(

    dataset_path,

    execution_plan

):

    pipeline_start = time.time()

    create_directories()

    experiment_dir = create_experiment_directory()

    print("=" * 60, flush=True)
    print("Starting preprocessing...", flush=True)

    start = time.time()

    prepared_data = preprocess_request(

        dataset_path,

        execution_plan

    )

    print(
        f"✓ Preprocessing completed in {time.time() - start:.2f} seconds",
        flush=True
    )

    print("=" * 60, flush=True)
    print("Starting model training...", flush=True)

    start = time.time()

    training_output = train_models(

        execution_plan,

        prepared_data

    )

    print(
        f"✓ Training completed in {time.time() - start:.2f} seconds",
        flush=True
    )

    best_pipeline = training_output[

        "best_model"

    ]

    print("=" * 60, flush=True)
    print("Saving best model...", flush=True)

    start = time.time()

    save_model(

        best_pipeline,

        experiment_dir

    )

    print(
        f"✓ Model saved in {time.time() - start:.2f} seconds",
        flush=True
    )

    print("=" * 60, flush=True)
    print("Generating visualizations...", flush=True)

    start = time.time()

    artifacts = generate_visualizations(

        execution_plan,

        training_output,

        prepared_data,

        experiment_dir

    )

    print(
        f"✓ Visualizations completed in {time.time() - start:.2f} seconds",
        flush=True
    )

    print("=" * 60, flush=True)
    print("Generating reports...", flush=True)

    start = time.time()

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

    print(
        f"✓ Reports generated in {time.time() - start:.2f} seconds",
        flush=True
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

    print("=" * 60, flush=True)
    print(
        f"✓ TOTAL PIPELINE TIME: {time.time() - pipeline_start:.2f} seconds",
        flush=True
    )
    print("=" * 60, flush=True)

    return response