# # import pandas as pd
# # import os
# # import matplotlib.pyplot as plt

# # from sklearn.model_selection import train_test_split
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler

# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier

# # from sklearn.metrics import (
# #     accuracy_score,
# #     precision_score,
# #     recall_score,
# #     f1_score
# # )

# # MODEL_REGISTRY = {
# #     "Logistic Regression": LogisticRegression(max_iter=1000),
# #     "Decision Tree": DecisionTreeClassifier(random_state=42),
# #     "Random Forest": RandomForestClassifier(random_state=42)
# # }


# # def train_pipeline(request_data):


# #     execution_plan = request_data["execution_plan"]
# #     dataset_path = request_data["dataset_path"]

# #     df = pd.read_csv(dataset_path)

# #     target = execution_plan["target_column"]

# #     X = df.drop(columns=[target])
# #     y = df[target]

# #     categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
# #     numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# #     # Preprocessing
# #     numeric_transformer = Pipeline([
# #         ("imputer", SimpleImputer(strategy="mean")),
# #         ("scaler", StandardScaler())
# #     ])

# #     categorical_transformer = Pipeline([
# #         ("imputer", SimpleImputer(strategy="most_frequent")),
# #         ("encoder", OneHotEncoder(handle_unknown="ignore"))
# #     ])

# #     preprocessor = ColumnTransformer([
# #         ("num", numeric_transformer, numerical_cols),
# #         ("cat", categorical_transformer, categorical_cols)
# #     ])

# #     # Train Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X,
# #         y,
# #         train_size=execution_plan["train_size"],
# #         random_state=42,
# #         stratify=y
# #     )


# #     artifact_dir = "/shared/artifacts"
# #     os.makedirs(artifact_dir, exist_ok=True)

# #     results = []

# #     best_pipeline = None
# #     best_accuracy = -1

# #     for model_info in execution_plan["recommended_models"]:

# #         model_name = model_info["name"]

# #         if model_name not in MODEL_REGISTRY:
# #             print(f"Skipping unsupported model: {model_name}")
# #             continue

# #         classifier = MODEL_REGISTRY[model_name]

# #         pipeline = Pipeline([
# #             ("preprocessor", preprocessor),
# #             ("classifier", classifier)
# #         ])

# #         pipeline.fit(X_train, y_train)

# #         y_pred = pipeline.predict(X_test)

# #         accuracy = accuracy_score(y_test, y_pred)

# #         precision = precision_score(
# #             y_test,
# #             y_pred,
# #             average="weighted",
# #             zero_division=0
# #         )

# #         recall = recall_score(
# #             y_test,
# #             y_pred,
# #             average="weighted",
# #             zero_division=0
# #         )

# #         f1 = f1_score(
# #             y_test,
# #             y_pred,
# #             average="weighted",
# #             zero_division=0
# #         )

# #         results.append({
# #             "model": model_name,
# #             "accuracy": round(float(accuracy), 4),
# #             "precision": round(float(precision), 4),
# #             "recall": round(float(recall), 4),
# #             "f1": round(float(f1), 4)
# #         })

# #         if accuracy > best_accuracy:
# #             best_accuracy = accuracy
# #             best_pipeline = pipeline

# #     results = sorted(
# #         results,
# #         key=lambda x: x["accuracy"],
# #         reverse=True
# #     )

# #     best_model = results[0]



# #     #plots
    
# #     model_names = [m["model"] for m in results]
# #     accuracies = [m["accuracy"] for m in results]

# #     plt.figure(figsize=(8, 5))

# #     bars = plt.bar(model_names, accuracies)

# #     plt.title("Model Accuracy Comparison")
# #     plt.ylabel("Accuracy")
# #     plt.ylim(0, 1)

# #     for bar, score in zip(bars, accuracies):
# #         plt.text(
# #             bar.get_x() + bar.get_width() / 2,
# #             score + 0.01,
# #             f"{score:.3f}",
# #             ha="center"
# #         )

# #     plt.tight_layout()

# #     accuracy_plot = os.path.join(
# #         artifact_dir,
# #         "accuracy_comparison.png"
# #     )

# #     plt.savefig(accuracy_plot)
# #     plt.close()

# #     return {

# #         "dataset_shape": list(df.shape),
# #         "problem_type": execution_plan["problem_type"],
# #         "target_column": target,

# #         "best_model": best_model,

# #         "all_models": results,

# #         "artifacts": {

# #             "accuracy_plot": accuracy_plot

# #         }

# #     }


# import os
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score
# )

# from models import MODEL_REGISTRY
# from plots import PLOT_REGISTRY


# ARTIFACT_DIR = "/shared/artifacts"


# def train_pipeline(request_data):

#     # Read Request

#     execution_plan = request_data["execution_plan"]
#     dataset_path = request_data["dataset_path"]

#     os.makedirs(ARTIFACT_DIR, exist_ok=True)

#     df = pd.read_csv(dataset_path)

#     target = execution_plan["target_column"]

#     X = df.drop(columns=[target])
#     y = df[target]

#     # Detect Feature Types

#     categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
#     numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

#     # Preprocessing Pipeline

#     numeric_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="mean")),
#         ("scaler", StandardScaler())
#     ])

#     categorical_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ])

#     preprocessor = ColumnTransformer([
#         ("num", numeric_transformer, numerical_cols),
#         ("cat", categorical_transformer, categorical_cols)
#     ])

#     # Train/Test Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         train_size=execution_plan["train_size"],
#         random_state=42,
#         stratify=y
#     )

#     # Train Models

#     results = []

#     trained_models = {}

#     predictions = {}

#     best_pipeline = None
#     best_accuracy = -1

#     for model_info in execution_plan["recommended_models"]:

#         model_name = model_info["name"]

#         if model_name not in MODEL_REGISTRY:

#             print(f"Skipping unsupported model {model_name}")

#             continue

#         classifier = MODEL_REGISTRY[model_name]

#         pipeline = Pipeline([

#             ("preprocessor", preprocessor),

#             ("classifier", classifier)

#         ])

#         pipeline.fit(X_train, y_train)

#         y_pred = pipeline.predict(X_test)

#         trained_models[model_name] = pipeline

#         predictions[model_name] = y_pred

#         accuracy = accuracy_score(y_test, y_pred)

#         precision = precision_score(
#             y_test,
#             y_pred,
#             average="weighted",
#             zero_division=0
#         )

#         recall = recall_score(
#             y_test,
#             y_pred,
#             average="weighted",
#             zero_division=0
#         )

#         f1 = f1_score(
#             y_test,
#             y_pred,
#             average="weighted",
#             zero_division=0
#         )

#         result = {

#             "model": model_name,

#             "accuracy": round(float(accuracy),4),

#             "precision": round(float(precision),4),

#             "recall": round(float(recall),4),

#             "f1": round(float(f1),4)

#         }

#         results.append(result)

#         if accuracy > best_accuracy:

#             best_accuracy = accuracy

#             best_pipeline = pipeline

#     # Sort Models
#     results = sorted(

#         results,

#         key=lambda x: x["accuracy"],

#         reverse=True

#     )

#     best_model = results[0]

#     best_model_name = best_model["model"]

#     best_predictions = predictions[best_model_name]

#     # Generate Visualizations

#     artifacts = {}

#     if "visualizations" in execution_plan:

#         for viz in execution_plan["visualizations"]:

#             viz_type = viz["type"]

#             if viz_type not in PLOT_REGISTRY:

#                 print(f"Visualization '{viz_type}' not supported")

#                 continue

#             artifact_path = PLOT_REGISTRY[viz_type](

#                 artifact_dir=ARTIFACT_DIR,

#                 results=results,

#                 model=trained_models[best_model_name],

#                 model_name=best_model_name,

#                 X_test=X_test,

#                 y_test=y_test,

#                 y_pred=best_predictions,

#                 preprocessor=preprocessor,

#                 options=viz

#             )

#             artifacts[viz_type] = artifact_path

#     # Return Response

#     return {

#         "dataset_shape": list(df.shape),

#         "problem_type": execution_plan["problem_type"],

#         "target_column": target,

#         "best_model": best_model,

#         "all_models": results,

#         "artifacts": artifacts

#     }