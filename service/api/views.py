import os
import re
from typing import Dict, Union

from fastapi import APIRouter, Body, FastAPI
from pydantic import BaseModel, Json

from service.log import app_logger

from ..ml_model import Model
from ..ner_model import NER


class CFG:
    num_workers = 8
    path = "ml/output_me5"
    config_path = os.path.join(path, "config.pth")
    model = "intfloat/multilingual-e5-large"
    gradient_checkpointing = False
    batch_size = 1
    target_cols = ["Исполнитель", "Группа тем", "Тема"]
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    max_len = 512


class RecoResponse(BaseModel):
    executor: Union[str, None]
    topic: Union[str, None]
    subtopic: Union[str, None]
    tags: Dict[str, object]


class RecoRequest(BaseModel):
    appeal: str
    confidenceThreshold: float


router = APIRouter()
ner = NER()
model = Model(CFG)


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.post("/reco")
async def get_reco(request: Json[RecoRequest] = Body()) -> RecoResponse:
    app_logger.info("REQUEST")

    appeal = request.appeal
    dt = ner.get_tags(appeal)
    appeal = "query: " + " ".join(re.findall(r"[а-яА-Я0-9 ёЁ\-\.,?!+a-zA-Z]+", appeal))

    threshold = request.confidenceThreshold
    str_exec_label, str_topic_label, str_subtopic_label = model.predict(appeal, threshold)

    respone = RecoResponse(executor=str_exec_label, topic=str_topic_label, subtopic=str_subtopic_label, tags=dt)
    return respone


def add_views(app: FastAPI) -> None:
    app.include_router(router)
