from typing import List
import asyncio

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from configs.config import AppConfig, ModelConfig
from infrastructure.models import BetterTransformerTextClassificationModel
from service.recognition import TextClassificationService
from handlers.recognition import PredictionHandler
from handlers.data_models import ResponseSchema


config = AppConfig.parse_file("./configs/app_config.yaml")
models = [
            BetterTransformerTextClassificationModel(conf.model, conf.model_path, conf.tokenizer)
            for conf in config.models
        ]

recognition_service = TextClassificationService(models)
recognition_handler = PredictionHandler(recognition_service, config.timeout)

app = FastAPI()
router = APIRouter()


@app.on_event("startup")
async def count_max_batch_size():
    print("Calculating Max batch size")
    batch_size = 100
    if models[0].device == 'cpu':
        app.max_batch_size = batch_size
    else:
        try:
            while True:
                text = ["this is simple text"] * batch_size
                inputs = [model.tokenize_texts(text) for model in models]
                outputs = [model(m_inputs) for model, m_inputs in zip(models, inputs)]
                batch_size += 100

        except RuntimeError as err:
            if "CUDA out of memory" in str(err):
                batch_size -= 100
                app.max_batch_size = batch_size
                print(f"Max batch size calculated = {app.max_batch_size}")

@app.on_event("startup")
def create_queues():
    app.models_queues = {}
    for md in models:
        task_queue = asyncio.Queue()
        app.models_queues[md.name] = task_queue
        asyncio.create_task(recognition_handler.handle(md.name, task_queue, app.max_batch_size))


@router.post("/process", response_model=ResponseSchema)
async def process(request: Request):
    text = (await request.body()).decode()

    results = []
    response_q = asyncio.Queue() # init a response queue for every request, one for all models
    for model_name, model_queue in app.models_queues.items():
        await model_queue.put((text, response_q))
        model_res = await response_q.get()
        results.append(model_res)
    return recognition_handler.serialize_answer(results)


app.include_router(router)


@app.get("/healthcheck")
async def main():
    return {"message": "I am alive"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="NLP Model Service",
        version="0.1.0",
        description="Inca test task",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get(
    "/documentation/swagger-ui/",
    response_class=HTMLResponse,
)
async def swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/documentation/openapi.json",
        title="API documentation"
    )


@app.get(
    "/documentation/openapi.json",
    response_model_exclude_unset=True,
    response_model_exclude_none=True,
)
async def openapi_endpoint():
    return custom_openapi()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=config.port, workers=config.workers)
