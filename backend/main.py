
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.title = "Logic API Matrix"
app.version = "0.1.0"
app.description = "API de logica para la solucion de problemas con matrices"

@app.get("/")
def root():
    return {"message": "Hello World" }

