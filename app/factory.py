from fastapi import FastAPI

from app.classifier import TriageClassifier
from app.schemas import KeywordResponse, TriageRequest, TriageResponse
from shared.vocabulary import KEYWORDS


def create_app(classifier: TriageClassifier) -> FastAPI:
    app = FastAPI(title="triAI-ml")

    @app.post("/classificar", response_model=TriageResponse)
    def classify(request: TriageRequest) -> TriageResponse:
        return classifier.classify(request)

    @app.get("/sintomas", response_model=list[KeywordResponse])
    def list_keywords() -> list[KeywordResponse]:
        return [
            KeywordResponse(code=keyword.code, term=keyword.term, label=keyword.label)
            for keyword in KEYWORDS
        ]

    return app

