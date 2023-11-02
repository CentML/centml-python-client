import os
import pickle
from http import HTTPStatus
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from centml.compiler.server_compilation import hidet_backend_server, storage_path, CompilationStatus, dir_cleanup
from centml.compiler import config_instance

app = FastAPI()


@app.get("/status/{model_id}")
async def status_handler(model_id: str):
    if not os.path.isdir(os.path.join(storage_path, model_id)):
        return {"status": CompilationStatus.NOT_FOUND}

    if not os.path.isfile(os.path.join(storage_path, model_id, "cgraph.pkl")):
        return {"status": CompilationStatus.COMPILING}

    if os.path.isfile(os.path.join(storage_path, model_id, "cgraph.pkl")):
        return {"status": CompilationStatus.DONE}

    # Something is wrong if we get here
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid status state")


@app.post("/compile_model/{model_id}")
async def compile_model_handler(model_id: str, model: UploadFile = File(...), inputs: UploadFile = File(...)):
    dir_cleanup(model_id)
    os.makedirs(os.path.join(storage_path, model_id))

    try:
        tfx_contents = await model.read()
        ei_contents = await inputs.read()
    except Exception as e:
        dir_cleanup(model_id)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error reading serialized content") from e
    finally:
        model.file.close()

    try:
        tfx_graph = pickle.loads(tfx_contents)
        example_inputs = pickle.loads(ei_contents)
    except Exception as e:
        dir_cleanup(model_id)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error loading pickled content") from e

    try:
        # This will save the cgraph to {storage_path}/{model_id}/cgraph.pkl
        hidet_backend_server(tfx_graph, example_inputs, model_id)
    except Exception as e:
        dir_cleanup(model_id)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Compilation Failed") from e


@app.get("/download/{model_id}")
async def download_handler(model_id: str):
    compiled_forward_path = os.path.join(storage_path, model_id, "cgraph.pkl")
    if not os.path.isfile(compiled_forward_path):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Compiled file not found")

    return FileResponse(compiled_forward_path)


def run():
    uvicorn.run("server:app", host=config_instance.SERVER_IP, port=int(config_instance.SERVER_PORT))


if __name__ == "__main__":
    run()
