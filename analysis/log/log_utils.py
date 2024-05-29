from typing import Dict, List
from os import getenv
from os.path import isfile, join
from csv import DictWriter

from analysis.config import (
    MODEL,
    N_CTX,
    N_GPU_LAYERS,
    N_BATCH,
)


def model_log(
    corpus: Dict[int, List[str]],
    pos_attributes: List[str],
    start_time: float,
    end_time: float,
) -> None:

    # Calculate number of calls
    calls = len(corpus) * len(pos_attributes)

    # Define the CSV file path
    csv_file_path = join(getenv("LOG_PATH"), "model_log.csv")

    # Define the headers for the CSV file
    headers = ["time", "n_gpu_layers", "n_batch", "n_ctx", "calls", "model"]

    # Create a list with the data to be written
    data = [
        {
            "time": end_time - start_time,
            "n_gpu_layers": N_GPU_LAYERS,
            "n_batch": N_BATCH,
            "n_ctx": N_CTX,
            "calls": calls,
            "model": MODEL,
        }
    ]

    # Check if the file exists
    file_exists = isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, "a", newline="") as file:
        writer = DictWriter(file, fieldnames=headers)

        # Write header only if the file is being created for the first time
        if not file_exists:
            writer.writeheader()

        # Append data to the CSV file
        writer.writerows(data)
