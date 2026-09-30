import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from app.classifier import TriageClassifier, celsius_to_fahrenheit
from app.factory import create_app
from shared.features import FEATURES
from shared.vocabulary import KEYWORDS


@pytest.fixture(scope="module")
def classifier(tmp_path_factory) -> TriageClassifier:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((300, len(FEATURES))), columns=FEATURES)
    target = rng.integers(1, 6, size=300)
    model = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(frame, target)

    path = tmp_path_factory.mktemp("model") / "model.pkl"
    joblib.dump({"model": model, "features": FEATURES}, path)
    return TriageClassifier(path)


@pytest.fixture(scope="module")
def client(classifier) -> TestClient:
    return TestClient(create_app(classifier))


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
    assert celsius_to_fahrenheit(None) is None


def test_classify_accepts_the_portuguese_contract_used_by_the_java_api(client):
    response = client.post(
        "/classificar",
        json={
            "idade": 54,
            "frequencia_cardiaca": 110,
            "pressao_sistolica": 95,
            "pressao_diastolica": 60,
            "saturacao_o2": 93,
            "temperatura": 38.4,
            "frequencia_respiratoria": 22,
            "dor": 7,
            "sexo_masculino": True,
            "queixa_principal": "Dor torácica",
            "ambulancia": True,
            "sintomas": ["chest-pain", "dyspnea"],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"corPrevista", "confianca"}
    assert body["corPrevista"] in {"Vermelho", "Laranja", "Amarelo", "Verde", "Azul"}
    assert 0 <= body["confianca"] <= 1


def test_classify_accepts_null_and_empty_payloads(client):
    assert client.post("/classificar", json={}).status_code == 200
    nulls = {"frequencia_cardiaca": 88, "dor": None, "sintomas": None, "queixa_principal": None}
    assert client.post("/classificar", json=nulls).status_code == 200


def test_classify_forces_red_on_critical_vitals(client):
    response = client.post("/classificar", json={"saturacao_o2": 80})

    assert response.json()["corPrevista"] == "Vermelho"


def test_classify_rejects_invalid_types(client):
    assert client.post("/classificar", json={"frequencia_cardiaca": "abc"}).status_code == 422


def test_keywords_endpoint_lists_the_whole_vocabulary(client):
    body = client.get("/sintomas").json()

    assert len(body) == len(KEYWORDS)
    assert set(body[0]) == {"codigo", "termo", "rotulo"}
    assert {"codigo": "chest-pain", "termo": "chest pain", "rotulo": "Dor torácica"} in body
