# Description

This repository makes use of [Python Bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python) to run open-source large language models (LLMs) locally. The `LLamaCpp` class from `langchain_community` dependency is used in `analysis/utils.py` to load models either downloaded to localhost or from [Hugging Face](https://huggingface.co/models). The installation uses GPU acceleration to make the analysis of millions of textual observations feasible. 

`main.py` is the entry point of the program and calls lower-level functions from `analysis/utils.py` or `analysis/database/db_utils.py`. Configuration constants are found in `analysis/config.py`.


Below are the steps of the analysis found in `main.py`:

1. Fetch all company reviews or other textual data from SQLite database found in `analysis/database/data/`.

    - From the list of `Review` objects returned a dictionary is created with review ids as keys and review text as values.


2. Most LLMs have a context window which will drop parts of longer textual observations. To account for the context window any longer company reviews are split into smaller parts containing full sentences which are then each passed to the model and scored individually and then aggregated into a total score using the mean later on.


3. Load model with `load_model()` function from `analysis/utils.py`. This function takes several constants defined in `analysis/config.py`.

    - `MODEL_PATH`: Path to model on localhost. Makes use of `MODEL_PATH` env variable of the same name.
    - `N_CTX`: Context length in tokens. Llama-2 supports 4096 tokens. 
    - `N_GPU_LAYERS`: Number of layers to offload to the GPUs. Check number of layers of chosen model. 
    - `N_BATCH`: Number of tokens to process in parallel. Should be a number between 1 and context length.

    - Other model parameters are abstracted away within the function:
        - `temperature=0`: Controls how stochastic output is. Deterministic outputs are chosen for reproducibility.
        - `max_tokens=10`: Maximum number of tokens returned.
        - `top_p=1`: Another parameter for random sampling. 1 is deterministic.


4. Generate scores for each company review or textual observation with the `generate_responses()` function found in `analysis/utils.py`. Function takes the `corpus` object, `DESCRIPTOR` constant, `POSITIVE_ATTRIBUTES` constant, both defined in `analysis/config.py`, a prompt function defined in `analysis/utils.py`, and `llm` model object defined in local scope. Function returns type `Dict[int, Dict[str, float]]` with outer keys as review ids and inner keys as structural aspects, such as centralization, and inner values as scores. 


5. Log model benchmarking to .csv file with `model_log()` function from `analysis/log/log_utils.py`.


6. Update database with similarity scores returned from the model for each structural aspect for each observation. `update_reviews_scores` function is found in `analysis/database/db_utils.py`.