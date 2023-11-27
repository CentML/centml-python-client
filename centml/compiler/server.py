import os
import pickle
from http import HTTPStatus
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from centml.compiler.server_compilation import hidet_backend_server, storage_path, CompilationStatus, dir_cleanup
from centml.compiler import config_instance

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


def get_status(model_id: str):
    if not os.path.isdir(os.path.join(storage_path, model_id)):
        return CompilationStatus.NOT_FOUND

    if not os.path.isfile(os.path.join(storage_path, model_id, "cgraph.zip")):
        return CompilationStatus.COMPILING

    if os.path.isfile(os.path.join(storage_path, model_id, "cgraph.zip")):
        return CompilationStatus.DONE

    return None


@app.get("/status/{model_id}")
async def status_handler(model_id: str):
    status = get_status(model_id)
    if status:
        return {"status": status}
    else:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Status check: invalid status state.")


def background_compile(model_id: str, model: UploadFile, inputs: UploadFile):
    try:
        tfx_contents = model.file.read()
        ei_contents = inputs.file.read()
    except Exception as e:
        logger.exception(f"Compilation: error reading serialized content. {e}")
        dir_cleanup(model_id)
        return
    finally:
        model.file.close()
        inputs.file.close()

    try:
        tfx_graph = pickle.loads(tfx_contents)
        example_inputs = pickle.loads(ei_contents)
    except Exception as e:
        logger.exception(f"Compilation: error loading pickled content. {e}")
        dir_cleanup(model_id)
        return

    try:
        # This will save the cgraph to {storage_path}/{model_id}/cgraph.zip
        hidet_backend_server(tfx_graph, example_inputs, model_id)
    except Exception as e:
        logger.exception(f"Compilation: error compiling model. {e}")
        dir_cleanup(model_id)


@app.post("/submit/{model_id}")
async def compile_model_handler(model_id: str, model: UploadFile, inputs: UploadFile, background_task: BackgroundTasks):
    status = get_status(model_id)
    if status is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Compilation: error checking status.")

    # Only compile if the model is not compiled or compiling
    if status != CompilationStatus.NOT_FOUND:
        return Response(status_code=200)

    # This effectively sets the model's status to COMPILING
    os.makedirs(os.path.join(storage_path, model_id))

    # perform the compilation in the background and return HTTP.OK to client
    background_task.add_task(background_compile, model_id, model, inputs)


@app.get("/download/{model_id}")
async def download_handler(model_id: str):
    compiled_forward_path = os.path.join(storage_path, model_id, "cgraph.zip")
    if not os.path.isfile(compiled_forward_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Download: compiled file not found")
    return FileResponse(compiled_forward_path)


def run():
    uvicorn.run("server:app", host=config_instance.SERVER_IP, port=int(config_instance.SERVER_PORT))


if __name__ == "__main__":
    run()
