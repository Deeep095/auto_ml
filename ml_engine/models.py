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

from sklearn.svm import (
    SVC,
    SVR
)

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


CLASSIFICATION_MODELS = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "Gradient Boosting": GradientBoostingClassifier(random_state=42),

    "AdaBoost": AdaBoostClassifier(random_state=42),

    "Support Vector Machine": SVC(probability=True),

    "KNN": KNeighborsClassifier(),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )

}


REGRESSION_MODELS = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(),

    "Lasso Regression": Lasso(),

    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),

    "Random Forest Regressor": RandomForestRegressor(random_state=42),

    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),

    "Support Vector Regressor": SVR(),

    "KNN Regressor": KNeighborsRegressor()

}


CLUSTERING_MODELS = {

    "KMeans": KMeans(
        n_clusters=3,
        random_state=42
    ),

    "DBSCAN": DBSCAN(),

    "Agglomerative Clustering": AgglomerativeClustering()

}


FORECASTING_MODELS = {

    # Reserved for future implementation

    "ARIMA": None,

    "SARIMA": None,

    "Prophet": None,

    "XGBoost Forecast": None

}


MODEL_REGISTRY = {

    **CLASSIFICATION_MODELS,

    **REGRESSION_MODELS,

    **CLUSTERING_MODELS,

    **FORECASTING_MODELS

}