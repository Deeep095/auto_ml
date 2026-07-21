import json
import joblib
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd

from auto_ml.ml_engine.config import (
    ARTIFACT_DIR,
    MODEL_DIR,
    REPORT_DIR,
    LOG_DIR,
    MODEL_FILENAME
)


def create_directories():

    directories = [

        ARTIFACT_DIR,

        MODEL_DIR,

        REPORT_DIR,

        LOG_DIR

    ]

    for directory in directories:

        Path(directory).mkdir(
            parents=True,
            exist_ok=True
        )


def create_experiment_directory():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = ARTIFACT_DIR / timestamp

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return experiment_dir


def save_json(data, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_json(path):

    with open(path, "r") as f:

        return json.load(f)


def save_dataframe(df, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )


def load_dataset(path):

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":

        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:

        return pd.read_excel(path)

    raise Exception(
        f"Unsupported dataset format : {suffix}"
    )


def save_model(model, experiment_dir):

    model_path = Path(experiment_dir) / MODEL_FILENAME

    joblib.dump(
        model,
        model_path
    )

    return model_path


def load_model(path):

    return joblib.load(path)


def copy_file(source, destination):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    return destination


def move_file(source, destination):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.move(
        source,
        destination
    )

    return destination


def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def experiment_name():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def file_exists(path):

    return Path(path).exists()


def directory_exists(path):

    return Path(path).exists() and Path(path).is_dir()


def delete_file(path):

    path = Path(path)

    if path.exists():

        path.unlink()


def delete_directory(path):

    path = Path(path)

    if path.exists():

        shutil.rmtree(path)


def list_files(directory):

    directory = Path(directory)

    if not directory.exists():

        return []

    return sorted(

        [

            str(file)

            for file in directory.iterdir()

            if file.is_file()

        ]

    )


def list_directories(directory):

    directory = Path(directory)

    if not directory.exists():

        return []

    return sorted(

        [

            str(folder)

            for folder in directory.iterdir()

            if folder.is_dir()

        ]

    )


def get_file_size(path):

    path = Path(path)

    if not path.exists():

        return 0

    return path.stat().st_size


def bytes_to_mb(size):

    return round(

        size / (1024 * 1024),

        2

    )


def artifact_summary(experiment_dir):

    experiment_dir = Path(experiment_dir)

    artifacts = []

    if not experiment_dir.exists():

        return artifacts

    for file in experiment_dir.iterdir():

        if file.is_file():

            artifacts.append({

                "name": file.name,

                "path": str(file),

                "size_mb": bytes_to_mb(

                    file.stat().st_size

                )

            })

    return artifacts


def build_response(

    dataset_shape,

    target_column,

    problem_type,

    best_model,

    all_models,

    artifacts

):

    return {

        "dataset_shape": dataset_shape,

        "problem_type": problem_type,

        "target_column": target_column,

        "best_model": best_model,

        "all_models": all_models,

        "artifacts": artifacts

    }