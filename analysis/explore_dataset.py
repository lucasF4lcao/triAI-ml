import pandas as pd

from shared.config import RAW_DATA_DIR

VITAL_COLUMNS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
INVESTIGATED_RANGES = {
    "temperature": (90, 110),
    "heartrate": (20, 250),
    "resprate": (5, 80),
    "o2sat": (50, 100),
    "sbp": (40, 300),
    "dbp": (20, 200),
}
EDSTAYS_CATEGORICALS = ["gender", "race", "arrival_transport", "disposition"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def describe_table(table: pd.DataFrame, name: str) -> None:
    section(f"TABELA {name.upper()}")
    print(f"Registros: {len(table)}")
    print(f"Colunas: {table.columns.tolist()}")
    print(f"Pacientes únicos: {table['subject_id'].nunique()}")
    print(f"Atendimentos únicos: {table['stay_id'].nunique()}")
    print(f"\nValores nulos:\n{table.isnull().sum()}")


def explore_triage(triage: pd.DataFrame) -> None:
    describe_table(triage, "triage")
    print(f"\nDistribuição de acuity:\n{triage['acuity'].value_counts(dropna=False).sort_index()}")
    print(f"\nEstatísticas dos sinais vitais:\n{triage[VITAL_COLUMNS].describe().T}")

    section("VALORES EXTREMOS")
    for column in VITAL_COLUMNS:
        print(f"{column}: menores {triage[column].nsmallest(5).tolist()} | maiores {triage[column].nlargest(5).tolist()}")

    section("FORA DAS FAIXAS INVESTIGADAS")
    for column, (low, high) in INVESTIGATED_RANGES.items():
        filled = triage[column].notna().sum()
        outside = ((triage[column] < low) | (triage[column] > high)).sum()
        print(f"{column}: {outside} de {filled} ({outside / filled:.2%})")

    section("SINAIS VITAIS POR ACUITY (MEDIANA)")
    print(triage.groupby("acuity")[VITAL_COLUMNS].median().round(2))


def explore_pain(triage: pd.DataFrame) -> None:
    section("COLUNA PAIN")
    numeric = pd.to_numeric(triage["pain"], errors="coerce")
    outside = ((numeric < 0) | (numeric > 10)).sum()

    print(f"Valores diferentes: {triage['pain'].nunique(dropna=False)}")
    print(f"Mais frequentes:\n{triage['pain'].value_counts(dropna=False).head(30)}")
    print(f"\nConvertidos para número: {numeric.notna().sum()}")
    print(f"Não nulos não convertidos: {triage['pain'].notna().sum() - numeric.notna().sum()}")
    print(f"Numéricos fora de 0-10: {outside} ({outside / numeric.notna().sum():.2%})")
    print(f"\nMediana de dor por acuity:\n{numeric.where(numeric.between(0, 10)).groupby(triage['acuity']).median()}")


def explore_edstays(edstays: pd.DataFrame, triage: pd.DataFrame) -> None:
    describe_table(edstays, "edstays")

    section("VARIÁVEIS CATEGÓRICAS DA EDSTAYS")
    for column in EDSTAYS_CATEGORICALS:
        print(f"\n{column}: {edstays[column].nunique()} categorias\n{edstays[column].value_counts()}")

    section("ACUITY POR FORMA DE CHEGADA (% por linha)")
    merged = triage[["stay_id", "acuity"]].merge(edstays[["stay_id", "arrival_transport"]], on="stay_id")
    print((pd.crosstab(merged["arrival_transport"], merged["acuity"], normalize="index") * 100).round(2))


def main() -> None:
    triage = pd.read_csv(RAW_DATA_DIR / "triage.csv.gz")
    edstays = pd.read_csv(RAW_DATA_DIR / "edstays.csv.gz")

    explore_triage(triage)
    explore_pain(triage)
    explore_edstays(edstays, triage)


if __name__ == "__main__":
    main()
