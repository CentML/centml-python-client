import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.post("/submit/{model_id}")
async def submit_handler(model_id: int):
	return {"message": "Compilation job submitted."}

@app.get("/status/{model_id}")
async def status_handler(model_id: int):
	return {"message": "Compilation job finished."}

@app.get("/download/{model_id}")
async def download_handler(model_id: int):
	return {"message": "Download successful."}

def run():
    uvicorn.run(app, host="0.0.0.0", port=8080)
