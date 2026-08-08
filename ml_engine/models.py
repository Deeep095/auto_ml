"""
Model registry: factories + hyperparameter schema.

This is the single source of truth for:
  1. The trainer   -> validates user params and instantiates estimators.
  2. The AI agent   -> reads the schema to know what is editable and
                       within which bounds (exposed via GET /models/schema).

Each entry in MODEL_SCHEMAS has:
    problem_type : one of classification / regression / clustering
    factory      : callable(**params) -> a fresh sklearn estimator
    params       : { param_name: spec } describing the editable knobs

A param spec looks like:
    {
        "type": "int" | "float" | "bool" | "choice",
        "default": <value>,
        "min": <number>,          # numeric only
        "max": <number>,          # numeric only
        "choices": [...],         # choice only
        "nullable": True,         # optional, allows None
        "description": "..."      # shown to the user / AI
    }
"""

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier
)

from sklearn.svm import SVC, SVR

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)


RANDOM_STATE = 42


def _num(kind, default, minimum, maximum, description, nullable=False):
    spec = {
        "type": kind,
        "default": default,
        "min": minimum,
        "max": maximum,
        "description": description
    }
    if nullable:
        spec["nullable"] = True
    return spec


def _choice(default, choices, description):
    return {
        "type": "choice",
        "default": default,
        "choices": choices,
        "description": description
    }


def _bool(default, description):
    return {
        "type": "bool",
        "default": default,
        "description": description
    }


MODEL_SCHEMAS = {

    # ---------------- Classification ----------------

    "Logistic Regression": {
        "problem_type": "classification",
        "factory": lambda **p: LogisticRegression(**p),
        "params": {
            "C": _num("float", 1.0, 0.001, 1000.0,
                      "Inverse regularization strength; smaller = stronger regularization."),
            "penalty": _choice("l2", ["l1", "l2", None],
                               "Regularization type."),
            "solver": _choice("lbfgs", ["lbfgs", "liblinear", "saga"],
                              "Optimization algorithm."),
            "max_iter": _num("int", 1000, 100, 10000,
                             "Maximum training iterations.")
        }
    },

    "Decision Tree": {
        "problem_type": "classification",
        "factory": lambda **p: DecisionTreeClassifier(random_state=RANDOM_STATE, **p),
        "params": {
            "max_depth": _num("int", None, 1, 100,
                              "Maximum tree depth; None = unlimited.", nullable=True),
            "min_samples_split": _num("int", 2, 2, 100,
                                      "Min samples required to split a node."),
            "min_samples_leaf": _num("int", 1, 1, 100,
                                     "Min samples required at a leaf."),
            "criterion": _choice("gini", ["gini", "entropy", "log_loss"],
                                "Split quality function.")
        }
    },

    "Random Forest": {
        "problem_type": "classification",
        "factory": lambda **p: RandomForestClassifier(random_state=RANDOM_STATE, **p),
        "params": {
            "n_estimators": _num("int", 100, 10, 1000,
                                "Number of trees in the forest."),
            "max_depth": _num("int", None, 1, 100,
                              "Maximum depth of each tree; None = unlimited.", nullable=True),
            "min_samples_split": _num("int", 2, 2, 100,
                                      "Min samples required to split a node."),
            "min_samples_leaf": _num("int", 1, 1, 100,
                                     "Min samples required at a leaf."),
            "max_features": _choice("sqrt", ["sqrt", "log2", None],
                                   "Features considered per split.")
        }
    },

    "Gradient Boosting": {
        "problem_type": "classification",
        "factory": lambda **p: GradientBoostingClassifier(random_state=RANDOM_STATE, **p),
        "params": {
            "n_estimators": _num("int", 100, 10, 1000,
                                "Number of boosting stages."),
            "learning_rate": _num("float", 0.1, 0.001, 1.0,
                                 "Shrinkage applied to each tree."),
            "max_depth": _num("int", 3, 1, 50,
                              "Maximum depth of each tree."),
            "subsample": _num("float", 1.0, 0.1, 1.0,
                             "Fraction of samples used per stage.")
        }
    },

    "AdaBoost": {
        "problem_type": "classification",
        "factory": lambda **p: AdaBoostClassifier(random_state=RANDOM_STATE, **p),
        "params": {
            "n_estimators": _num("int", 50, 10, 1000,
                                "Number of weak learners."),
            "learning_rate": _num("float", 1.0, 0.001, 2.0,
                                 "Weight applied to each classifier.")
        }
    },

    "Naive Bayes": {
        "problem_type": "classification",
        "factory": lambda **p: GaussianNB(**p),
        "params": {
            "var_smoothing": _num("float", 1e-9, 1e-12, 1e-3,
                                 "Variance smoothing for stability.")
        }
    },

    "Support Vector Machine": {
        "problem_type": "classification",
        "factory": lambda **p: SVC(**p),
        "params": {
            "C": _num("float", 1.0, 0.001, 1000.0,
                      "Regularization strength."),
            "kernel": _choice("rbf", ["linear", "poly", "rbf", "sigmoid"],
                             "Kernel function."),
            "gamma": _choice("scale", ["scale", "auto"],
                            "Kernel coefficient."),
            "probability": _bool(False,
                                "Enable probability estimates (needed for ROC).")
        }
    },

    "KNN": {
        "problem_type": "classification",
        "factory": lambda **p: KNeighborsClassifier(**p),
        "params": {
            "n_neighbors": _num("int", 5, 1, 100,
                               "Number of neighbours."),
            "weights": _choice("uniform", ["uniform", "distance"],
                              "Neighbour weighting scheme."),
            "p": _choice(2, [1, 2],
                        "Distance metric: 1=Manhattan, 2=Euclidean.")
        }
    },

    "Neural Network": {
        "problem_type": "classification",
        "factory": lambda **p: MLPClassifier(random_state=RANDOM_STATE, **p),
        "params": {
            "hidden_layer_sizes": _choice((100,), [(50,), (100,), (100, 50), (128, 64)],
                                         "Hidden layer architecture."),
            "activation": _choice("relu", ["relu", "tanh", "logistic"],
                                 "Activation function."),
            "alpha": _num("float", 0.0001, 1e-6, 1.0,
                         "L2 regularization term."),
            "max_iter": _num("int", 500, 100, 5000,
                            "Maximum training iterations.")
        }
    },

    # ---------------- Regression ----------------

    "Linear Regression": {
        "problem_type": "regression",
        "factory": lambda **p: LinearRegression(**p),
        "params": {
            "fit_intercept": _bool(True, "Whether to fit an intercept term.")
        }
    },

    "Ridge Regression": {
        "problem_type": "regression",
        "factory": lambda **p: Ridge(random_state=RANDOM_STATE, **p),
        "params": {
            "alpha": _num("float", 1.0, 0.0001, 1000.0,
                         "Regularization strength.")
        }
    },

    "Lasso Regression": {
        "problem_type": "regression",
        "factory": lambda **p: Lasso(random_state=RANDOM_STATE, **p),
        "params": {
            "alpha": _num("float", 1.0, 0.0001, 1000.0,
                         "Regularization strength.")
        }
    },

    "Decision Tree Regressor": {
        "problem_type": "regression",
        "factory": lambda **p: DecisionTreeRegressor(random_state=RANDOM_STATE, **p),
        "params": {
            "max_depth": _num("int", None, 1, 100,
                              "Maximum tree depth; None = unlimited.", nullable=True),
            "min_samples_split": _num("int", 2, 2, 100,
                                      "Min samples required to split a node."),
            "min_samples_leaf": _num("int", 1, 1, 100,
                                     "Min samples required at a leaf.")
        }
    },

    "Random Forest Regressor": {
        "problem_type": "regression",
        "factory": lambda **p: RandomForestRegressor(random_state=RANDOM_STATE, **p),
        "params": {
            "n_estimators": _num("int", 100, 10, 1000,
                                "Number of trees in the forest."),
            "max_depth": _num("int", None, 1, 100,
                              "Maximum depth of each tree; None = unlimited.", nullable=True),
            "min_samples_split": _num("int", 2, 2, 100,
                                      "Min samples required to split a node."),
            "min_samples_leaf": _num("int", 1, 1, 100,
                                     "Min samples required at a leaf.")
        }
    },

    "Gradient Boosting Regressor": {
        "problem_type": "regression",
        "factory": lambda **p: GradientBoostingRegressor(random_state=RANDOM_STATE, **p),
        "params": {
            "n_estimators": _num("int", 100, 10, 1000,
                                "Number of boosting stages."),
            "learning_rate": _num("float", 0.1, 0.001, 1.0,
                                 "Shrinkage applied to each tree."),
            "max_depth": _num("int", 3, 1, 50,
                              "Maximum depth of each tree."),
            "subsample": _num("float", 1.0, 0.1, 1.0,
                             "Fraction of samples used per stage.")
        }
    },

    "Support Vector Regressor": {
        "problem_type": "regression",
        "factory": lambda **p: SVR(**p),
        "params": {
            "C": _num("float", 1.0, 0.001, 1000.0,
                      "Regularization strength."),
            "kernel": _choice("rbf", ["linear", "poly", "rbf", "sigmoid"],
                             "Kernel function."),
            "gamma": _choice("scale", ["scale", "auto"],
                            "Kernel coefficient."),
            "epsilon": _num("float", 0.1, 0.0, 10.0,
                           "Epsilon-tube within which no penalty is given.")
        }
    },

    "KNN Regressor": {
        "problem_type": "regression",
        "factory": lambda **p: KNeighborsRegressor(**p),
        "params": {
            "n_neighbors": _num("int", 5, 1, 100,
                               "Number of neighbours."),
            "weights": _choice("uniform", ["uniform", "distance"],
                              "Neighbour weighting scheme."),
            "p": _choice(2, [1, 2],
                        "Distance metric: 1=Manhattan, 2=Euclidean.")
        }
    },

    # ---------------- Clustering ----------------

    "KMeans": {
        "problem_type": "clustering",
        "factory": lambda **p: KMeans(random_state=RANDOM_STATE, n_init=10, **p),
        "params": {
            "n_clusters": _num("int", 3, 2, 50,
                              "Number of clusters to form.")
        }
    },

    "DBSCAN": {
        "problem_type": "clustering",
        "factory": lambda **p: DBSCAN(**p),
        "params": {
            "eps": _num("float", 0.5, 0.01, 100.0,
                       "Max distance between two neighbouring samples."),
            "min_samples": _num("int", 5, 1, 100,
                               "Min samples in a neighbourhood for a core point.")
        }
    },

    "Agglomerative Clustering": {
        "problem_type": "clustering",
        "factory": lambda **p: AgglomerativeClustering(**p),
        "params": {
            "n_clusters": _num("int", 3, 2, 50,
                              "Number of clusters to form."),
            "linkage": _choice("ward", ["ward", "complete", "average", "single"],
                              "Linkage criterion.")
        }
    }

}


# Names grouped by problem type — convenient for menus / the AI agent.
MODELS_BY_PROBLEM_TYPE = {}
for _name, _spec in MODEL_SCHEMAS.items():
    MODELS_BY_PROBLEM_TYPE.setdefault(
        _spec["problem_type"], []
    ).append(_name)


def model_exists(name):
    return name in MODEL_SCHEMAS


def get_default_params(name):
    """Return {param: default} for a model."""
    return {
        param: spec["default"]
        for param, spec in MODEL_SCHEMAS[name]["params"].items()
    }


def _coerce_and_validate(name, raw_params):
    """
    Validate a raw {param: value} dict against the model's schema.

    - Unknown params are dropped (with a warning).
    - Types are coerced (int/float/bool).
    - Numeric values are clamped to [min, max].
    - choice values must be in the allowed set, else the default is used.
    - None is allowed only when the spec is nullable.

    Returns (clean_params, warnings).
    """
    schema = MODEL_SCHEMAS[name]["params"]
    clean = {}
    warnings = []

    for key, value in (raw_params or {}).items():

        if key not in schema:
            warnings.append(f"Unknown param '{key}' ignored for {name}.")
            continue

        spec = schema[key]

        if value is None:
            if spec.get("nullable"):
                clean[key] = None
            else:
                warnings.append(f"'{key}' cannot be null; using default.")
                clean[key] = spec["default"]
            continue

        kind = spec["type"]

        try:
            if kind == "int":
                value = int(value)
            elif kind == "float":
                value = float(value)
            elif kind == "bool":
                value = bool(value)
        except (TypeError, ValueError):
            warnings.append(
                f"'{key}'={value} is not a valid {kind}; using default."
            )
            clean[key] = spec["default"]
            continue

        if kind in ("int", "float"):
            low, high = spec["min"], spec["max"]
            if value < low:
                warnings.append(f"'{key}' clamped up to {low}.")
                value = low
            elif value > high:
                warnings.append(f"'{key}' clamped down to {high}.")
                value = high

        if kind == "choice" and value not in spec["choices"]:
            warnings.append(
                f"'{key}'={value} not in {spec['choices']}; using default."
            )
            value = spec["default"]

        clean[key] = value

    return clean, warnings


def build_estimator(name, raw_params=None):
    """
    Return (estimator, warnings) for a model name and user-supplied params.
    Params are validated/clamped against the schema before instantiation.
    """
    if name not in MODEL_SCHEMAS:
        raise ValueError(f"Unknown model: {name}")

    clean, warnings = _coerce_and_validate(name, raw_params)
    estimator = MODEL_SCHEMAS[name]["factory"](**clean)
    return estimator, warnings


def public_schema():
    """
    Serializable view of the registry (no factories/lambdas) for the
    GET /models/schema endpoint and the AI agent's context.
    """
    out = {}
    for name, spec in MODEL_SCHEMAS.items():
        out[name] = {
            "problem_type": spec["problem_type"],
            "params": spec["params"]
        }
    return out

