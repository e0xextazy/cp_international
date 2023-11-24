from typing import Dict, List

from fastapi import APIRouter, Body, FastAPI
from pydantic import BaseModel, Json

from service.log import app_logger

from ..ner_model import NER


class RecoResponse(BaseModel):
    executor: str | None
    topic: str | None
    subtopic: str | None
    tags: Dict[str, List[str]]


class RecoRequest(BaseModel):
    appeal: str
    confidenceThreshold: float


router = APIRouter()
ner = NER()


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
    print(appeal)
    threshold = request.confidenceThreshold
    print(threshold)

    dt = {
        "LOC": ["Москва"],
        "ORG": ["МЧС"],
        "PER": ["Иван", "Иванов"],
        "PHONE": [],
        "MONEY": [],
        "ADDRESS": ["ул. Энтузиастов"],
        "DATE": ["09.09.09"],
    }
    respone = RecoResponse(executor="executor_1", topic=None, subtopic=None, tags=dt)
    return respone


def add_views(app: FastAPI) -> None:
    app.include_router(router)
