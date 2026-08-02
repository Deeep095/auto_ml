# 🤖 Virtuoso AutoML

Virtuoso AutoML is a production-grade **Telegram-based AutoML Assistant** that enables users to build and evaluate machine learning models without writing code.

Users simply upload a dataset through Telegram, and Virtuoso automatically:

- 📂 Uploads and manages datasets
- 📊 Analyzes the data
- 🤖 Recommends suitable ML models
- ⚙️ Allows manual customization
- 🧠 Trains multiple machine learning models
- 📈 Compares model performance
- 📑 Generates reports and visualizations
- 📲 Sends results directly back to Telegram

---

# 🚀 System Architecture

It follows a containerized microservice architecture.

```
                    Telegram
                        │
                        ▼
               n8n Telegram Workflow
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Session Management             Analyze & Train Workflow
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
             FastAPI ML Engine
                        │
                        ▼
          Scikit-Learn Processing
                        │
                        ▼
     Reports • Models • Visualizations
                        │
                        ▼
                 Telegram Response
```

---

# 🏗️ Components

## 1. Telegram Agent (n8n)

A stateful n8n workflow responsible for

- User interaction
- Session management
- Dataset handling
- State transitions
- Conversation routing
- Model customization
- Triggering model training

---

## 2. FastAPI ML Engine

A Python backend responsible for

- Dataset preprocessing
- Model training
- Model evaluation
- Artifact generation
- Report generation

---

## 3. Shared Storage

Docker volumes are shared between n8n and the ML Engine.

```
/shared/uploads
/shared/artifacts
/shared/reports
/shared/models
```

This allows

- uploaded datasets
- trained models
- generated reports
- plots

to be accessed by both services.

---

# 🧠 Machine Learning Pipeline

The FastAPI backend automatically performs the entire ML lifecycle.

---

## Data Preprocessing

The preprocessing engine automatically

- Detects numerical and categorical features
- Handles missing values
- Performs feature encoding
- Applies feature scaling
- Splits the dataset into Train/Test sets

Supported preprocessing includes

- Mean Imputation
- Median Imputation
- Most Frequent Imputation
- OneHot Encoding
- Ordinal Encoding
- Standard Scaling
- MinMax Scaling
- Robust Scaling

---

# 🤖 Supported Models

## Classification

- Random Forest
- Gradient Boosting
- Logistic Regression
- Decision Tree
- AdaBoost
- Naive Bayes
- Support Vector Machine
- KNN
- Neural Network

---

## Regression

- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Support Vector Regressor
- KNN Regressor

---

# 📊 Evaluation Metrics

Depending on the problem type, It computes

Classification

- Accuracy
- Precision
- Recall
- F1 Score

Regression

- MAE
- RMSE
- MAPE
- R² Score

---

# 📈 Generated Artifacts

Model automatically generates

- Confusion Matrix
- ROC Curve
- Feature Importance
- HTML Report
- JSON Metrics
- Summary Report
- Serialized Best Model

---

# 🔀 n8n Workflow

The Telegram automation is split into **two workflows**.

---

# 1️⃣ Main Router Workflow

The Main Router manages the entire conversation using deterministic routing based on user session state.

## Routing Flow

| Current State | Route | Description |
|---------------|-------|-------------|
| idle | show_menu | Display main menu |
| upload_ds | prompt_upload | Ask user to upload CSV |
| waiting_for_previous_selection | list_previous | Show previous datasets |
| dataset_menu | select_previous | Select existing dataset |
| dataset_menu | handle_upload | Handle uploaded dataset |
| guess_plan | train_recommended | Recommend default training plan |
| waiting_target_column | customize_start | Start customization |
| waiting_problem_type | pick_target | User selects target column |
| waiting_models | pick_problem_type | Select Classification / Regression |
| waiting_models | toggle_model | Toggle selected models |
| confirm_train | models_done | Finalize selected models |
| training | confirm_yes | Trigger Analyze & Train Workflow |

---

## Conversation Flow

```
idle
 │
 ▼
show_menu

 ├── upload
 │      │
 │      ▼
 │ prompt_upload
 │      │
 │      ▼
 │ handle_upload
 │      │
 │      ▼
 │ dataset_menu
 │
 └── previous
        │
        ▼
list_previous
        │
        ▼
select_previous
        │
        ▼
dataset_menu
```

---

## Training Flow

```
dataset_menu
      │
      ▼
train
      │
      ▼
guess_plan
      │
      ▼
train_recommended
      │
      ▼
customize
      │
      ▼
waiting_target_column
      │
      ▼
pick_target
      │
      ▼
waiting_problem_type
      │
      ▼
pick_problem_type
      │
      ▼
waiting_models
      │
      ├──── toggle_model
      │
      └──── models_done
               │
               ▼
         confirm_train
               │
               ▼
          confirm_yes
               │
               ▼
            training
               │
               ▼
            Results
```

---

# 2️⃣ Analyze & Train Workflow

After confirmation, a second workflow is executed.

It performs

- HTTP POST request to FastAPI
- Model training
- Response parsing
- Markdown formatting
- Sending metrics
- Sending generated plots

---

# 📂 Project Structure

```
Virtuoso_Project/

│
├── auto_ml/
│   ├── app.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   ├── plots.py
│   ├── reports.py
│
├── shared/
│   ├── uploads/
│   ├── artifacts/
│   ├── reports/
│   └── models/
│
├── n8n_workflows/
│   ├── main_workflow.json
│   └── analyze_train_workflow.json
│
├── docker-compose.yml
│
└── README.md
```

---

# 🛠️ Local Setup

## Prerequisites

- Docker
- Docker Compose
- n8n
- Telegram Bot Token

---

## Start the Environment

```bash
docker compose up --build -d
```

The FastAPI server will be available inside Docker.

---

## Configure Telegram

Create a Telegram bot using **BotFather** and configure its token inside your n8n Telegram credentials.

---

## Local Webhook (Optional)

If running n8n locally, expose it using ngrok.

```bash
ngrok http 5678
```

Update the Telegram Trigger webhook URL with the generated HTTPS URL.

---

# 📥 Importing n8n Workflows

To import the workflows

1. Open n8n.
2. Click **Add Workflow**.
3. Select **Import from File**.
4. Import

```
main_workflow.json
```

and

```
analyze_train_workflow.json
```

5. Configure your Telegram credentials.


---

# 🚀 Future Improvements

- Hyperparameter Optimization
- Cross Validation
- User driven customization in model training 
- Explainable AI (SHAP)
- Time Series Forecasting
- XGBoost
- LightGBM
- CatBoost
- PDF Report Generation
- Model Deployment
- Interactive Dashboard

---

# 🛠️ Tech Stack

- Python
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Docker
- n8n
- Telegram Bot API

---

