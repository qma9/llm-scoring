from datetime import timedelta
from time import perf_counter

from analysis.log import setup_logging, logger, model_log
from analysis.database import get_all_reviews, update_reviews_scores
from analysis.utils import (
    load_model,
    prompt_general,
    generate_responses,
    split_long_string,
)
from analysis.config import (
    CHARACTER_MAX,
    MODEL,
    MODEL_PATH,
    N_CTX,
    N_GPU_LAYERS,
    N_BATCH,
    DESCRIPTOR,
    POSITIVE_ATTRIBUTES,
)


def main() -> None:

    total_start_time = perf_counter()  # Benchmarking total runtime

    # Fetch all reviews from database
    reviews = (
        get_all_reviews()
    )  # TESTING, in get_all_reviews() change fetch back to .all() for production
    logger.info(f"Total reviews: {len(reviews)}", extra={"total_reviews": len(reviews)})

    # Create a dictionary with review_id as keys and review_text as values
    reviews_dict = {review.review_id: review.review_text for review in reviews}

    # Split reviews longer than context maximum into sentences
    corpus = split_long_string(reviews_dict, CHARACTER_MAX)

    # Load model
    llm = load_model(MODEL_PATH, N_CTX, N_GPU_LAYERS, N_BATCH)

    model_start_time = perf_counter()  # benchmarking model runtime

    # Generate ranking scores
    responses = generate_responses(
        corpus, DESCRIPTOR, POSITIVE_ATTRIBUTES, prompt_general, llm
    )

    model_end_time = perf_counter()  # benchmarking model runtime

    # Model log
    model_log(corpus, POSITIVE_ATTRIBUTES, model_start_time, model_end_time)

    # Print model run time
    model_runtime = timedelta(seconds=(model_end_time - model_start_time))
    logger.info(f"{MODEL} model ran in: {model_runtime}")

    # Update structural aspect scores in database
    update_reviews_scores(reviews, responses)

    total_end_time = perf_counter()  # Benchmarking total runtime

    # Log time
    total_runtime = timedelta(seconds=(total_end_time - total_start_time))
    logger.info(
        f"Analysis complete. Total runtime: {total_runtime}",
        extra={
            "total_runtime": total_runtime,
            "model_runtime": model_runtime,
            "model": MODEL,
            "corpus_length": len(reviews),
        },
    )


if __name__ == "__main__":

    # Setup logging
    listener = setup_logging()

    # Call main()
    main()

    # Stop listener
    listener.stop()
