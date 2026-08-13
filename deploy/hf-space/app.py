"""Kuber.ai thermal-surrogate demo — FastAPI server (run on vicpc).

Loads the trained surface model once at startup, serves the PhysicsX-style
frontend and two JSON endpoints:
  GET /api/cases          -> list of in-distribution preset geometries
  GET /api/predict/{i}    -> full predicted field + CFD ground truth + metrics

Run (from repo root, pnemo env):
  python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from infer import Engine

HERE = Path(__file__).resolve().parent
CKPT = os.environ.get("KUBER_CKPT", "outputs/geot_medium_surface.pt")
DATA = os.environ.get("KUBER_DATA", "/home/shubhj/simshift/npz")
SPLITS = os.environ.get("KUBER_SPLITS", "/home/shubhj/simshift/npz/splits.json")
ENG: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] loading model + dataset (one-time)...", flush=True)
    ENG["e"] = Engine(ckpt=CKPT, data=DATA, splits=SPLITS)
    print(f"[app] ready: device={ENG['e'].dev} cases={len(ENG['e'].ids)}", flush=True)
    yield
    ENG.clear()


app = FastAPI(title="Kuber.ai Thermal Surrogate", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/cases")
def cases():
    return ENG["e"].list_cases()


@app.get("/api/info")
def info():
    return {"device": ENG["e"].device_label()}


@app.get("/api/geometry/{i}")
def geometry(i: int):
    return ENG["e"].geometry(int(i))


@app.get("/api/stl/{i}")
def stl_case(i: int):
    body = ENG["e"].stl_case(int(i))
    cid = ENG["e"].ids[int(i)]
    return Response(content=body, media_type="model/stl",
                    headers={"Content-Disposition": f'attachment; filename="kuber_{cid}.stl"'})


@app.get("/api/predict/{i}")
def predict(i: int):
    return JSONResponse(ENG["e"].predict(int(i)))


@app.get("/api/predict_custom")
def predict_custom(fins: int = 8, length: float = None, width: float = None,
                   height1: float = None, height2: float = None, gap: float = None,
                   thickness_fins: float = None, solidTemp: float = None, envTemp: float = None):
    ov = dict(fins=fins, length=length, width=width, height1=height1, height2=height2,
              gap=gap, thickness_fins=thickness_fins, solidTemp=solidTemp, envTemp=envTemp)
    return JSONResponse(ENG["e"].predict_custom(ov))


@app.get("/api/predict_cp")
def predict_cp(L: float = 0.12, W: float = 0.02, H: float = 0.003, q: float = 2e5,
               u_in: float = 1.0, T_in: float = 300.0, fluid: str = "water"):
    ov = dict(L=L, W=W, H=H, q=q, u_in=u_in, T_in=T_in, fluid=fluid)
    return JSONResponse(ENG["e"].predict_custom_cp(ov))


@app.get("/api/export")
def export(fmt: str, fins: int = 8, length: float = None, width: float = None,
           height1: float = None, height2: float = None, gap: float = None,
           thickness_fins: float = None, solidTemp: float = None):
    ov = dict(fins=fins, length=length, width=width, height1=height1, height2=height2,
              gap=gap, thickness_fins=thickness_fins, solidTemp=solidTemp)
    body, mime, fname = ENG["e"].export(ov, fmt)
    return Response(content=body, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
