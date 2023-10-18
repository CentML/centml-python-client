import torch
import sys
import os
import shutil
import centml
import multiprocessing as mp
from centml.compiler.server import run
import time
    

# Recursively deletes all the directories/files in directory_path
def clear_server_storage():
    directory_path = os.path.join(os.getcwd(), "centml/compiler/pickled_objects_server/")

    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    if not os.path.isdir(directory_path):
        print(f"'{directory_path}' is not a directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def child_process(model):
    model_compiled_child = torch.compile(model, backend="centml")
    model_compiled_child(torch.randn(1, 3, 224, 224))
    print("CHILD DONE")

def double_compilation_test():
    model = torch.hub.load(
        'pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False
    )
    model = model.eval()

    mp.set_start_method('forkserver', force=True)
    child = mp.Process(target=child_process, args=(model,))

    child.start()

    model_compiled = torch.compile(model, backend="centml")
    model_compiled(torch.randn(1, 3, 224, 224))

    print("PARENT DONE")

    child.join()

def speed_test():
    model = torch.hub.load(
        'pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False
    )
    model = model.eval()

    def get_avg_time(func):
        times = []
        for i in range(100):
            start_time = time.time()
            func()
            times.append(time.time() - start_time)
        return sum(times)/len(times)

    def uncompiled_wrapper():
        model(torch.randn(1, 3, 224, 224))

    model_compiled = torch.compile(model, backend="centml")

    def compiled_wrapper():
        model_compiled(torch.randn(1, 3, 224, 224))

    # since the first call to the compiled model will call torch.compile, do it now to avoid an outlier in the times
    model_compiled(torch.randn(1, 3, 224, 224))
    print("uncompiled avg time:", get_avg_time(uncompiled_wrapper))
    print("compiled avg time:", get_avg_time(compiled_wrapper))

def find_max_diff(a, b):
    diff = torch.abs(a - b)
    max_diff = torch.max(diff)
    return max_diff.item()

def check_diff():
    model = torch.hub.load(
        'pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False
    )
    model = model.eval()

    model_custom = torch.compile(model, backend="centml")

    for i in range(10):
        x = torch.randn(1, 3, 224, 224)
        compiled_y = model_custom(x)
        uncompiled_y = model(x)

        if (dif := find_max_diff(compiled_y, uncompiled_y)) > 1e-5:
            print("DIFFERENCE TOO LARGE")
        else:
            print("DIFFERENCE OK:", dif)

def main():
    print(os.getcwd())

    model = torch.hub.load(
        'pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False
    )
    model = model.eval()

    model_custom = torch.compile(model, backend="centml")

    for i in range(10):
        model_custom(torch.randn(1, 3, 224, 224))

if __name__ == '__main__':
    args = sys.argv[1:]
    if "--clear" in args:
        clear_server_storage()
    if "--double" in args:
        double_compilation_test()
    elif "--time" in args:
        speed_test()
    elif "--check_diff" in args:
        check_diff()
    else:
        main()