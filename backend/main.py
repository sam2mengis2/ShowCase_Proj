from enum import Enum
from fastapi import FastAPI

class ModelType(str, Enum):
    brain = "brain"
    other = "other"
    world = "world"


app = FastAPI()


@app.get("/model/{model}")
async def fn(model: str, bar: int):
    return {"message": model, "bar": bar}


@app.get("/")
async def run():
    return {"message": "Server has started"}


@app.get("/model/{model_name}")
async def get_model(model_name: ModelType):
    message = {"model_name": model_name, "message": ""}
    if model_name is ModelType.brain:
        message["message"] = "your brain is cooked!"
    elif model_name is ModelType.other:
        message["message"] = "bro is not himself"
    else:
        message["message"] = "Hello, World?"

    return message
