from langchain_community.llms.llamacpp import LlamaCpp
from numpy import array, isnan, nan, sum as np_sum

from typing import List, Dict

from analysis.log import logger
from analysis.config import MODEL


def load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
) -> LlamaCpp:
    """
    Load Llama model.
    """
    Llama_model = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=5,
        top_p=1,
        verbose=True,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
    )

    logger.info(
        f"Loaded {MODEL} model.",
        extra={
            "temperature": 0,
            # LLama-2 uses SentencePiece Byte-Pair tokenization, after testing with LlamaTokenizerFast and "hf-internal-testing/llama-tokenizer",
            # float output, e.g. "-0.67" or "0.45", are 6 tokens long
            "max_tokens": 6,
            "top_p": 1,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
        },
    )
    return Llama_model


def split_long_string(reviews: Dict[int, str], n: int) -> Dict[int, List[str]]:
    """
    Split longer documents into shorter paragraphs to fit within maximum context length.
    """
    outer_result = {}

    for id, review in reviews.items():

        # Split the string into sentences based on periods
        sentences = review.split(".")

        # Initialize result list
        inner_result = []

        # Iterate through sentences and build shorter strings
        current_string = ""
        for sentence in sentences:
            if len(current_string) + len(sentence) + 1 <= n:
                # Add the sentence to the current string if it doesn't exceed the limit
                current_string += sentence + "."
            else:
                # If the current string exceeds the limit, start a new one
                inner_result.append(current_string.strip())
                current_string = sentence + "."

        # Add the last string to the result
        if current_string:
            inner_result.append(current_string.strip())

        outer_result[id] = inner_result

    logger.info("Split documents to fit max context length.")
    return outer_result


def generate_responses(
    reviews: Dict[int, List[str]],
    descriptor: str,
    attributes: List[str],
    prompt_template: callable,
    llm: LlamaCpp,
) -> Dict[int, Dict[str, float]]:
    """
    Generate scores for each company review or textual observation.
    Function returns type Dict[int, Dict[str, float]] with outer keys as review ids and inner keys as structural aspects, such as centralization, and inner values as scores.
    """

    scores = {}

    for key, value in reviews.items():

        inner_dict = {}

        for attribute in attributes:

            inner_scores = []

            for string in value:

                prompt = prompt_template(string, descriptor, attribute)

                response = llm.invoke(prompt)

                try:
                    response_float = float(response.strip())
                except ValueError as e:
                    response_float = nan
                    logger.error(
                        f"Unable to convert model response to float: {e}",
                        extra={
                            "document_id": key,
                            "document_text": value,
                            "document_part": string,
                            "model_response": response.strip(),
                            "error": e,
                        },
                    )

                inner_scores.append(response_float)

            inner_scores_array = array(inner_scores)
            filtered_inner_scores = inner_scores_array[~isnan(inner_scores_array)]

            inner_dict[attribute] = np_sum(filtered_inner_scores) / len(
                filtered_inner_scores
            )

        scores[key] = inner_dict

    logger.info(
        f"Generated scores for {descriptor}s.",
        extra={"prompt_function": prompt_template},
    )
    return scores


def prompt_general(text: str, descriptor: str, concept: str) -> str:
    """
    Uncovering semantics of concepts using GPT-4 and other LLMs
    (Le Mens et al., 2023)
    General Prompt used for [INST] [/INST] part.
    """

    prompt = f"""
    <<SYS>>\n
    Your only purpose is to provide a measure of typicality, semantic similarity, of {descriptor}s to the concept of {concept}. 
    Provide your response as a score between -1.00 and 1.00, where -1.00 means ‘not typical at all’, 0.00 means ‘neutral’, and 1.00 means ‘extremely typical’.
    Strictly only respond with a floating point number with a precision of 2, e.g. "-0.23" or "0.67" and with no other text.
    
    Example responses:
    [INST]
    Here’s a company review:\nStrong leadership is a highlight, but opportunities for employees to contribute to decision-making processes are limited.\n
    How typical is this company review of centralization?
    [/INST]
    0.85

    [INST]
    Here’s a company review:\nLeadership values input from all levels. Decision-making is distributed, allowing lower-level employees to influence company-wide decisions.\n
    How typical is this company review of centralization?
    [/INST]
    -0.95

    \n<</SYS>>\n

    [INST]
    Here’s a {descriptor}:\n{text}\n
    How typical is this {descriptor} of {concept}?
    [/INST]
    """

    return prompt


# Prompt template for company reviews, only positive side of sprectrum
def review_prompt_single(review: str, attribute: str) -> str:

    prompt = f"""
    <<SYS>>\n
    You are an assistant who scores company reviews with a score of 0.0 to 1.0.

    You can only answer with a score between 0.0 and 1.0 with only one decimal place, for example 0.6 with no other text at all.

    Your score must represent how much the given review suggests that the company exhibits a specific cultural attribute.

    The closer to 1.0 your float response is, the more strongly the given review suggests that the company exhibits the specified cultural attribute.

    The closer to 0.5 your float response is, the more neutral the given review is for suggesting that the company exhibits the specified cultural attribute.

    The closer to 0.0 your float response is, the less strongly the given review suggests that the company exhibits the specified cultural attribute.

    Previous example assistant answers to reviews:

    [INST]
    Does the following company review suggests the company exhibits a centralized culture?
    The review: "The executives hold all the final decision-making power within the company, leaving no decisions to lower management."
    [/INST]
    0.9

    [INST] 
    Does the following company review suggests the company exhibits an autonomous culture?
    The review: "Employees are given specific processes to follow when completing their work, giving them no freedom to chose how to get things done."
    [/INST]
    0.1
    \n<</SYS>>\n\n

    [INST]
    Does the following company review suggests the company exhibits a(n) {attribute} culture?
    The review: {review}
    Assistant:
    [/INST]
    """

    return prompt


def review_prompt_comp0(review: str, pos_attribute: str, neg_attribute: str) -> str:
    """
    Prompt template for company reviews, comprehensive view.
    """

    prompt = f"""
    <<SYS>>\n
    You are an assistant who scores company reviews and can only answer with a score between -1.0 and 1.0 with only one decimal place, for example 0.6 with no other text at all.

    Your score must represent how much the given review suggests that the company exhibits a specific cultural attribute or its antonym.

    The closer to 1.0 your score is, the more strongly the given review suggests that the company exhibits the specified cultural attribute and not its antonym.

    The closer to 0.0 your score is, the more neutral the given review is between the specific cultural attribute and its antonym.

    The closer to -1.0 your score, the less strongly the given review suggests that the company exhibits the specified cultural attribute and its antonym.

    Previous assistant scores to reviews:

    [INST]
    Does the following company review suggests the company exhibits a centralized or decentralized culture?
    The review: "The executives hold all the final decision-making power within the company, leaving no decisions to lower management."
    [/INST]
    0.9

    [INST] 
    Does the following company review suggests the company exhibits an autonomous culture?
    The review: "Employees are given specific processes to follow when completing their work, giving them no freedom to chose how to get things done."
    [/INST]
    -0.9
    \n<</SYS>>\n\n

    [INST]
    Does the following company review suggests the company exhibits a(n) {pos_attribute} or {neg_attribute} culture?
    The review: {review}
    Assistant:
    [/INST]
    """

    return prompt


def review_prompt_comp1(review: str, pos_attribute: str, neg_attribute: str) -> str:
    """
    Prompt template for company reviews, comprehensive view.
    """

    definitions = {
        "centralized": "(of an activity or organization) controlled by a single authority or managed in one place.",
        "decentralized": "(of an activity or organization) controlled by several managers or authorities rather than one single one.",
    }

    prompt = f"""
    <<SYS>>\n
    You are an assistant who scores company reviews and can only answer with a score between -1.0 and 1.0 with only one decimal place, for example "0.6" or "-0.6" with no other text.

    Your score must represent how much the given review suggests that the company exhibits a specific cultural attribute or its opposite.

    The definition of {pos_attribute} (positive attribute) is: {definitions[pos_attribute]}.

    The definition of {neg_attribute} (negative attribute) is: {definitions[neg_attribute]}. 

    The closer to 1.0 your score is, the more strongly the given review suggests that the company exhibits a(n) {pos_attribute} culture (positive).

    The closer to -1.0 your score is, the more strongly the given review suggests that the company exhibits a(n) opposite {neg_attribute} culture (negative).

    The closer to 0.0 your score is, the more neutral the given review is between a(n) {pos_attribute} or {neg_attribute} culture (neutral).

    Previous assistant scores to reviews:

    [INST]
    Does the following company review suggests the company exhibits a centralized or decentralized culture?
    The review: "The executives hold all the final decision-making power within the company, leaving no decisions to lower management."
    [/INST]
    1.0

    [INST] 
    Does the following company review suggests the company exhibits a centralized or decentralized culture?
    The review: "Low level managers have the power to make final decisions on projects and don't have to ask the executives to sign off on decisions ever."
    [/INST]
    -1.0
    \n<</SYS>>\n\n

    [INST]
    Does the following company review suggests the company exhibits a(n) {pos_attribute} or {neg_attribute} culture?
    The review: {review}
    Assistant:
    [/INST]
    """

    return prompt


# Prompt template for earning calls
def ecalls_template(review: str, attribute: str) -> str:
    """
    Earnings call prompt template.
    """

    prompt = f"""
    <<SYS>>\n
    You are an assistant who scores earning call transcripts with a score of 0.0 to 1.0.

    You can only answer with a score between 0.0 and 1.0 with only one decimal place, for example 0.6 with no other text at all.

    Your score must represent how much the given transcript suggests that the company exhibits a specific cultural attribute.

    The closer to 1.0 your float response is, the more strongly the given transcript suggests that the company exhibits the specified cultural attribute.

    The closer to 0.5 your float response is, the more neutral the given transcript is for suggesting that the company exhibits the specified cultural attribute.

    The closer to 0.0 your float response is, the less strongly the given transcript suggests that the company exhibits the specified cultural attribute.

    Previous example assistant answers to reviews:

    [INST]
    Does the following transcript suggests the company exhibits a centralized culture?
    The transcript: "The executives hold all the final decision-making power within the company, leaving no decisions to lower management."
    [/INST]
    0.9

    [INST] 
    Does the following transcript suggests the company exhibits an autonomous culture?
    The transcript: "Employees are given specific processes to follow when completing their work, giving them no freedom to chose how to get things done."
    [/INST]
    0.1
    \n<</SYS>>\n\n

    [INST]
    Does the following transcript suggests the company exhibits a(n) {attribute} culture?
    The transcript: {review}
    Assistant:
    [/INST]
    """

    return prompt


def dummy_prompt(review: str, pos_attribute: str, neg_attribute: str) -> str:
    """
    Dummy variable prompt.
    """

    prompt = f"""
    <s>[INST]<<SYS>>
    You can only answer with the following characters: 0, 1, or 2.
    I am going to show you a company review, and you will answer according to what the review suggests about the company's organizational culture.
    Answer with a 0 if the review suggests the company's organizational culture is more {pos_attribute} than {neg_attribute}.
    Answer with a 2 if the review suggests the company's organizational culture is more {neg_attribute} than {pos_attribute}.
    Answer with a 1 if the review is neutral about the company's organizational culture being more {pos_attribute} or {neg_attribute}.
    <</SYS>>
    The review: "{review}"
    Your answer:[/INST]

    <s>[INST]<<SYS>>
    Answer with the following character: N.
    <</SYS>>[/INST]
    """

    return prompt


if __name__ == "__main__":
    pass
