from pydantic import BaseModel, ConfigDict, Field


class TriageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    heart_rate: int | None = Field(None, alias="frequencia_cardiaca")
    systolic_pressure: int | None = Field(None, alias="pressao_sistolica")
    diastolic_pressure: int | None = Field(None, alias="pressao_diastolica")
    oxygen_saturation: int | None = Field(None, alias="saturacao_o2")
    temperature: float | None = Field(None, alias="temperatura")
    respiratory_rate: int | None = Field(None, alias="frequencia_respiratoria")
    pain: int | None = Field(None, alias="dor")
    chief_complaint: str | None = Field(None, alias="queixa_principal")
    keywords: list[str] | None = Field(None, alias="sintomas")
    ambulance: bool | None = Field(None, alias="ambulancia")
    is_male: bool | None = Field(None, alias="sexo_masculino")


class TriageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    predicted_color: str = Field(alias="corPrevista")
    confidence: float = Field(alias="confianca")


class KeywordResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(alias="codigo")
    term: str = Field(alias="termo")
    label: str = Field(alias="rotulo")
