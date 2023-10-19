import uvicorn
from typing import Annotated
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pickle
import os
from centml.compiler.dynamo_server import hidet_backend_server, storage_path, Compilation_Status, dir_cleanup

app = FastAPI()

count = 0

@app.get("/status/{model_id}")
async def status_handler(model_id: str):

    if not os.path.isdir(os.path.join(storage_path, model_id)):
        return {"status": Compilation_Status.NOT_FOUND}

    if not os.path.isfile(os.path.join(storage_path, model_id, "cgraph.pkl")):
        return {"status": Compilation_Status.COMPILING}

    if os.path.isfile(os.path.join(storage_path, model_id, "cgraph.pkl")):
        return {"status": Compilation_Status.DONE}
    
    # Something is wrong if we get here
    raise HTTPException(status_code=400, detail="Invalid status state")


@app.post("/compile_model/")
async def compile_model_handler(model_id: Annotated[str, Form()], serialized_model: UploadFile = File(...), serialized_example_inputs: UploadFile = File(...)):

    # Leave the directory empty until compilation complete.
    os.makedirs(os.path.join(storage_path, model_id))

    try:
        tfx_contents = await serialized_model.read()
        ei_contents = await serialized_example_inputs.read()
    except Exception:
        dir_cleanup(model_id)
        raise HTTPException(status_code=400, detail="Error reading serialized content")
    finally:
        serialized_model.file.close()
    
    try:
        tfx_graph = pickle.loads(tfx_contents)
        example_inputs = pickle.loads(ei_contents)
    except Exception:
        dir_cleanup(model_id)
        raise HTTPException(status_code=400, detail="Error loading pickled content")	
    
    try:
        # This will safe the model to {storage_path}/{model_id}/cgraph.pkl
        hidet_backend_server(tfx_graph, example_inputs, model_id)
    except Exception:
        dir_cleanup(model_id)
        raise HTTPException(status_code=400, detail="Compilation Failed")


@app.get("/download/{model_id}")
async def download_handler(model_id: str):
    compiled_forward_path = os.path.join(storage_path, model_id, "cgraph.pkl")
    if not os.path.isfile(compiled_forward_path):
        raise HTTPException(status_code=400, detail="Compiled file not found")
    
    return FileResponse(compiled_forward_path)


def run():
    server_IP = os.getenv('CENTML_SERVER_IP', default="0.0.0.0")
    server_port = os.getenv('CENTML_SERVER_PORT', default="8080")
    uvicorn.run("server:app", host=server_IP, port=server_port, reload=True)

if __name__ == "__main__":
    run()