import io
import os
from http import HTTPStatus
import logging
import uvicorn
import torch
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from centml.compiler.server_compilation import hidet_backend_server
from centml.compiler.utils import dir_cleanup
from centml.compiler.config import config_instance, CompilationStatus
from centml.compiler.utils import get_server_compiled_forward_path

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=config_instance.MINIMUM_GZIP_SIZE)  # type: ignore


def get_status(model_id: str):
    if not os.path.isdir(os.path.join(config_instance.SERVER_BASE_PATH, model_id)):
        return CompilationStatus.NOT_FOUND

    if not os.path.isfile(get_server_compiled_forward_path(model_id)):
        return CompilationStatus.COMPILING

    return CompilationStatus.DONE


@app.get("/status/{model_id}")
async def status_handler(model_id: str):
    status = get_status(model_id)
    if status:
        return {"status": status}
    else:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Status check: invalid status state.")


def background_compile(model_id: str, tfx_graph, example_inputs):
    try:
        compiled_graph_module = hidet_backend_server(tfx_graph, example_inputs)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Compilation: error compiling model. {e}")
        dir_cleanup(model_id)

    try:
        save_path = get_server_compiled_forward_path(model_id)
        torch.save(compiled_graph_module, save_path, pickle_protocol=config_instance.PICKLE_PROTOCOL)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Saving graph module failed: {e}")


def read_upload_files(model_id: str, model: UploadFile, inputs: UploadFile):
    try:
        tfx_contents = io.BytesIO(model.file.read())
        ei_contents = io.BytesIO(inputs.file.read())
    except Exception as e:
        dir_cleanup(model_id)
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Compilation: error reading serialized content: {e}"
        ) from e
    finally:
        model.file.close()
        inputs.file.close()

    try:
        tfx_graph = torch.load(tfx_contents)
        example_inputs = torch.load(ei_contents)
    except Exception as e:
        dir_cleanup(model_id)
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Compilation: error loading content with torch.load: {e}"
        ) from e

    return tfx_graph, example_inputs


@app.post("/submit/{model_id}")
async def compile_model_handler(model_id: str, model: UploadFile, inputs: UploadFile, background_task: BackgroundTasks):
    status = get_status(model_id)
    if status is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Compilation: error checking status.")

    # Only compile if the model is not compiled or compiling
    if status != CompilationStatus.NOT_FOUND:
        return Response(status_code=200)

    # This effectively sets the model's status to COMPILING
    os.makedirs(os.path.join(config_instance.SERVER_BASE_PATH, model_id))

    tfx_graph, example_inputs = read_upload_files(model_id, model, inputs)

    # perform the compilation in the background and return HTTP.OK to client
    background_task.add_task(background_compile, model_id, tfx_graph, example_inputs)


@app.get("/download/{model_id}")
async def download_handler(model_id: str):
    compiled_forward_path = get_server_compiled_forward_path(model_id)
    if not os.path.isfile(compiled_forward_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Download: compiled file not found")
    return FileResponse(compiled_forward_path)


def run():
    uvicorn.run(app, host=config_instance.SERVER_IP, port=int(config_instance.SERVER_PORT))


if __name__ == "__main__":
    run()
