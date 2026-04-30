
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import logic


app = FastAPI(
    title="Logic API Matrix",
    version="0.1.0",
    description="API de logica para la solucion de problemas con matrices",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatrixPairPayload(BaseModel):
    matrix_a: List[List[float]]
    matrix_b: List[List[float]]


class MatrixPayload(BaseModel):
    matrix: List[List[float]]


class SystemPayload(BaseModel):
    matrix_a: List[List[float]]
    vector_b: List[float]


def _execute(operation):
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def root():
    return {
        "message": "API de algebra lineal activa",
        "routes": [
            "/matrices/suma",
            "/matrices/resta",
            "/matrices/multiplicacion",
            "/matrices/determinante",
            "/matrices/inversa",
            "/matrices/gauss",
            "/matrices/gauss-jordan",
            "/matrices/traspuesta",
            "/matrices/cramer",
        ],
    }


@app.post("/matrices/suma")
def suma(payload: MatrixPairPayload):
    return _execute(lambda: logic.sum_matrices(payload.matrix_a, payload.matrix_b))


@app.post("/matrices/resta")
def resta(payload: MatrixPairPayload):
    return _execute(lambda: logic.subtract_matrices(payload.matrix_a, payload.matrix_b))


@app.post("/matrices/multiplicacion")
def multiplicacion(payload: MatrixPairPayload):
    return _execute(lambda: logic.multiply_matrices(payload.matrix_a, payload.matrix_b))


@app.post("/matrices/determinante")
def determinante(payload: MatrixPayload):
    return _execute(lambda: logic.determinant(payload.matrix))


@app.post("/matrices/inversa")
def inversa(payload: MatrixPayload):
    return _execute(lambda: logic.inverse(payload.matrix))


@app.post("/matrices/gauss")
def gauss(payload: SystemPayload):
    return _execute(lambda: logic.gauss(payload.matrix_a, payload.vector_b))


@app.post("/matrices/gauss-jordan")
def gauss_jordan(payload: SystemPayload):
    return _execute(lambda: logic.gauss_jordan(payload.matrix_a, payload.vector_b))


@app.post("/matrices/traspuesta")
def traspuesta(payload: MatrixPayload):
    return _execute(lambda: logic.transpose_matrix(payload.matrix))


@app.post("/matrices/cramer")
def cramer(payload: SystemPayload):
    return _execute(lambda: logic.cramer(payload.matrix_a, payload.vector_b))

