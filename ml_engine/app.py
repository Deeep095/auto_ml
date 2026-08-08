from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import traceback

from pipeline import run_pipeline
from models import (
    public_schema,
    MODELS_BY_PROBLEM_TYPE,
    model_exists,
    get_default_params
)


app = FastAPI(
    title="AutoML Engine",
    version="1.1.0"
)


# ----------------------------
# Request / plan schema
# ----------------------------

class ModelSelection(BaseModel):
    name: str
    priority: Optional[int] = None
    # Per-model hyperparameters (validated/clamped in the engine).
    params: Optional[Dict[str, Any]] = None


class Visualization(BaseModel):
    type: str


class Preprocessing(BaseModel):
    missing_values: str = "mean"
    categorical_missing: str = "most_frequent"
    encoding: str = "onehot"
    scaling: str = "standard"


class ExecutionPlan(BaseModel):
    problem_type: str
    target_column: str
    recommended_models: List[ModelSelection] = Field(default_factory=list)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    visualizations: List[Visualization] = Field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    confidence: float = 1.0


class TrainRequest(BaseModel):
    dataset_path: str
    execution_plan: ExecutionPlan


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def home():
    return {
        "message": "AutoML Engine",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models/schema")
def models_schema(problem_type: Optional[str] = None):
    """
    Editable hyperparameter schema per model.

    Consumed by n8n and the AI agent to know which values a user may
    customize (name, type, default, valid range/choices) before training.
    Optional ?problem_type=classification filters the result.
    """
    schema = public_schema()

    if problem_type:
        schema = {
            name: spec
            for name, spec in schema.items()
            if spec["problem_type"] == problem_type
        }

    return {
        "models_by_problem_type": MODELS_BY_PROBLEM_TYPE,
        "schema": schema
    }


@app.get("/models/{model_name}/defaults")
def model_defaults(model_name: str):
    """Default params for a single model (handy for 'reset to default')."""
    if not model_exists(model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}"
        )
    return {
        "model": model_name,
        "defaults": get_default_params(model_name)
    }


@app.post("/train")
def train(request: TrainRequest):

    print("========== REQUEST RECEIVED ==========", flush=True)

    try:
        # Pydantic already validated the shape; hand the plan to the pipeline
        # as a plain dict so the rest of the engine stays unchanged.
        result = run_pipeline(
            request.dataset_path,
            request.execution_plan.model_dump()
        )

        print("========== REQUEST SUCCESS ==========", flush=True)
        print(result["best_model"], flush=True)
        return result

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing field : {str(e)}"
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
