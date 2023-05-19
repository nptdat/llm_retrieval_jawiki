import os
from pathlib import Path
from logging import basicConfig
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from qa.schemas import QueryRequest, QueryResponse, QAConfig
from .answerer import Answerer


basicConfig(level="INFO")


config_path = Path("../config")
cfg_qa = QAConfig.load(config_path)

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN") or cfg_qa.bearer_token
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


async def get_answerer():
    return Answerer(config_path)


app = FastAPI(dependencies=[Depends(validate_token)])


@app.post("/query")
async def query(request: QueryRequest):
    try:
        answer = answerer.answer(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get("/ping")
def ping():
    return {"result": "Health!"}


@app.on_event("startup")
async def startup():
    global answerer
    answerer = await get_answerer()

def start():
    uvicorn.run("qa.answer.api:app", host="0.0.0.0", port=8000, reload=True)
