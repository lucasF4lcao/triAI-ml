from app.classifier import TriageClassifier
from app.factory import create_app

app = create_app(TriageClassifier())
