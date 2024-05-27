import os
import json
from pathlib import Path

import neptune
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetHandleByIndex,
    NVMLError,
)


def init_neptune(config={}, api_token=None):
    run = None

    if config:
        if api_token:
            os.environ["NEPTUNE_API_TOKEN"] = api_token

        workspace = config["workspace"]
        project_name = config["project-name"]
        project = f"{workspace}/{project_name}"

        if project not in neptune.management.get_project_list():
            project = neptune.management.create_project(
                name=project_name,
                workspace=workspace,
                visibility="priv",
            )

        run = neptune.init_run(
            project=project,
            name=config["run-name"],
            tags=config["tags"],
        )

    return run


def get_gpu_info(mem_unit="GB"):
    device_info = []
    mem_factor = {"GB": 1024**3, "MB": 1024**2, "KB": 1024}

    try:
        nvmlInit()

        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            device = nvmlDeviceGetName(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)

            device_info.append(
                {
                    "device": device,
                    "memory": {
                        f"total ({mem_unit})": round(
                            mem_info.total / mem_factor[mem_unit], 3
                        ),
                        f"free ({mem_unit})": round(
                            mem_info.free / mem_factor[mem_unit], 3
                        ),
                        f"used ({mem_unit})": round(
                            mem_info.used / mem_factor[mem_unit], 3
                        ),
                    },
                }
            )

    except NVMLError as error:
        print(repr(error), "Likely no GPU available. Continuing anyway.")

    return device_info


def save_and_upload_output(
    output,
    run=None,
    save_file="output/evaluate.json",
    json_cls=None,
):
    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True)
    with open(save_file, "w") as f:
        json.dump(output, f, indent=2, sort_keys=False, cls=json_cls)

    if run is not None:
        run[f"reference/{save_file.name}"].upload(str(save_file))
