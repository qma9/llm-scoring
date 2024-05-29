from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import ValidationError

from contextlib import contextmanager
from typing import List, Dict
from dotenv import load_dotenv
from os import environ

from ..log import logger

from .models import Review, Company
from .base_models import ReviewBase

# Load environment variables
load_dotenv()


# Create the engine and session
engine = create_engine(environ.get("URL_DB"), connect_args={"check_same_thread": False})
SessionFactory = sessionmaker(autocommit=False, bind=engine)


@contextmanager
def get_db():
    """
    Get a database session.
    """
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


def get_all_reviews() -> List[Review]:
    """
    Get all company reviews from database.
    """

    with get_db() as session:
        reviews = session.query(
            Review
        ).all()  # .filter(Review.employer_id == 7633).limit(100).all() for testing

    return reviews


def update_reviews_scores(
    reviews: List[Review], similarity_scores_dict: Dict[int, Dict[str, float]]
) -> None:
    """
    Update reviews with similarity scores.
    """

    with get_db() as session:
        # Update the Review instances
        for review in reviews:
            try:
                # Get the Review instance from the current session or the database
                current_review = (
                    session.query(Review)
                    .filter(Review.review_id == review.review_id)
                    .first()
                )

                # Get the similarity scores for the current review
                similarity_scores = similarity_scores_dict.get(review.review_id)

                # Update the Review instance with the similarity scores
                if similarity_scores:
                    for attribute, score in similarity_scores.items():
                        setattr(current_review, attribute, score)
            except AttributeError:
                print(
                    f"\nReview with ID {review.review_id} not found in the database.\n"
                )
                logger.error(
                    f"Review with ID {review.review_id} not found in the database."
                )
                continue

        # Commit the changes
        session.commit()
        logger.info("Committed review scores to database.")


def get_all_companies():
    """
    Get all companies from database that have reviews.
    Check which companies have reviews.
    --------------------------

    Note: Not a production function.
    """

    with get_db() as session:
        companies = session.query(Company).join(Review).distinct().all()

    return companies


def update_employer_name():
    """
    Update employer names for companies scraped during testing.
    Bug was fixed that was overriding employer names with None during scraping reviews.
    --------------------------

    Note: Not a production function.
    """
    id_to_name = {
        10085: "WidePoint",
        20011: "Ziegler Inc.",
        476062: "ADS",
        1149: "Astronics",
        4: "AAR",
        2592: "RadNet",
        6092: "Alexion Pharmaceuticals",
        15: "AMD",
        7633: "NVIDIA",
        143310: "Chili's Grill and Bar",
        40772: "Meta",
        1138: "Apple",
        9079: "Google",
        1651: "Microsoft",
    }

    with get_db() as session:

        companies = (
            session.query(Company)
            .filter(Company.employer_id.in_(id_to_name.keys()))
            .all()
        )

        for company in companies:
            company.employer_name = id_to_name[company.employer_id]

        session.commit()
