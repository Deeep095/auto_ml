from fastapi import FastAPI
from fastapi import HTTPException
import traceback
from auto_ml.ml_engine.pipeline import run_pipeline


app = FastAPI(
    title="Virtuoso AutoML Engine",
    version="1.0.0"
)


@app.get("/")
def home():

    return {

        "message": "Virtuoso AutoML Engine",

        "status": "running"

    }


@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


@app.post("/train")
def train(request: dict):

    try:

        dataset_path = request["dataset_path"]

        execution_plan = request["execution_plan"]

        result = run_pipeline(

            dataset_path,

            execution_plan

        )

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