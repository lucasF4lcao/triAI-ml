# triAI-ml

Serviço de classificação de triagem (Protocolo de Manchester) usado pelo `triAI-api-java`. Treinado com o MIMIC-IV-ED. É um serviço interno: quem fala FHIR com os hospitais é a API Java.

## Como funciona

Um HistGradientBoosting prediz o **ESI (1 a 5)** registrado pelos enfermeiros do MIMIC-IV-ED a partir de 296 features:

- sinais vitais: temperatura, frequência cardíaca, frequência respiratória, saturação de O2, pressão sistólica e diastólica;
- dor (0 a 10), chegada de ambulância ou helicóptero e sexo;
- 287 palavras-chave de queixa principal (palavras e bigramas em inglês, do MIMIC), cada uma com rótulo em português.

Valores ausentes são tratados pelo próprio modelo, sem imputação. O ESI vira cor de Manchester por equivalência ordinal (1 Vermelho, 2 Laranja, 3 Amarelo, 4 Verde, 5 Azul), e a cor é forçada para **Vermelho** quando há sinais vitais críticos: SpO2 < 90, PAS < 80, FC fora de 40-150, FR fora de 8-30 ou temperatura fora de 35-40,5 °C. É uma adaptação metodológica, não uma equivalência clínica perfeita (concordância ESI x MTS de κ = 0,51 na literatura).

## Vocabulário de queixas

`shared/vocabulary.json` é a fonte única da lista de palavras-chave (código, termo em inglês, rótulo em português). O treino, a inferência e o `GET /sintomas` leem esse arquivo. Mudar a lista exige retreinar.

A queixa chega de três formas, combinadas: códigos em `sintomas`, texto livre em inglês e texto livre em português (casado pelo rótulo, sem acentos). Um código de bigrama ativa também as suas palavras, como acontece no texto de treino.

## Estrutura

```
app/        API FastAPI: fábrica (factory), schemas e inferência (classifier)
shared/     usado pelo treino e pela API: config, features, vocabulário e Manchester
training/   dataset (limpeza e features) e train (treino e avaliação)
analysis/   exploração do dataset e anotações
tests/      testes (pytest)
models/     modelo treinado (gerado)
data/       MIMIC-IV-ED (nunca versionado, dado licenciado)
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # testes
```

Coloque o MIMIC-IV-ED em `data/mimic-iv-ed-2.2/ed/` (ou crie um link simbólico).

## Treinar

Rode a partir da raiz do repositório:

```
python -m training.train   # gera models/triage_model.pkl
```

O treino usa cerca de 3 GB de memória no pico. Em máquinas com pouca RAM, rode com um limite de memória para não derrubar o sistema.

## Rodar a API

```
uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
```

**`POST /classificar`**: todos os campos são opcionais, e a temperatura vai em °C.

| Campo | Tipo |
|---|---|
| `frequencia_cardiaca`, `frequencia_respiratoria` | inteiro |
| `pressao_sistolica`, `pressao_diastolica` | inteiro |
| `saturacao_o2` | inteiro |
| `temperatura` | decimal, em °C |
| `dor` | inteiro de 0 a 10 |
| `sintomas` | lista de códigos do vocabulário |
| `queixa_principal` | texto, em inglês ou português |
| `ambulancia`, `sexo_masculino` | booleano |

Devolve `{"corPrevista": "Laranja", "confianca": 0.71}`. Campos desconhecidos são ignorados.

**`GET /sintomas`**: devolve `[{"codigo": "chest-pain", "termo": "chest pain", "rotulo": "Dor torácica"}, ...]`.

Atenção: sem nenhum sinal vital, o modelo tende a devolver Vermelho, porque no MIMIC a ausência de sinais costuma indicar paciente em estado crítico. A API Java recusa Bundles sem sinais vitais reconhecidos para não repassar esse resultado.

## Testes

```
python -m pytest
```

## Métricas

Teste de 20% (83.620 casos); validação cruzada de 5 folds com acurácia de 0,686 ± 0,006. Nenhuma etapa usa o alvo para preencher dados.

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| ESI 1 (Vermelho) | 0,75 | 0,63 | 0,69 |
| ESI 2 (Laranja) | 0,67 | 0,57 | 0,62 |
| ESI 3 (Amarelo) | 0,70 | 0,84 | 0,76 |
| ESI 4 (Verde) | 0,57 | 0,16 | 0,25 |
| ESI 5 (Azul) | 0,11 | 0,07 | 0,09 |

Acurácia de 68,9%, e 98,8% dos casos saem a no máximo um nível de ESI do valor real.

## Limitações conhecidas

- **Subtriagem.** O modelo maximiza a acurácia e favorece o Amarelo: 39% dos casos Laranja e 12% dos Vermelho do teste saíram como Amarelo ou Azul. A regra de sinais vitais críticos só protege os extremos. Se a segurança pesar mais que a acurácia, reponderar as classes troca acurácia por recall dos casos graves.
- **Verde e Azul** têm recall baixo (0,16 e 0,07).
- **Dados americanos.** O ESI do MIMIC é rotulado por enfermeiros dos EUA, e a lista de palavras-chave vem de texto em inglês. O desempenho num hospital brasileiro não está medido. Queixas em português fora da lista só contam pelo casamento dos rótulos.
- **Sem idade.** O MIMIC-IV-ED não traz a idade (está em outra tabela, fora do acesso atual), então ela não entra no modelo.
- As métricas medem o ESI. A cor devolvida pela API pode virar Vermelho pela regra de sinais vitais críticos.

## Deploy no Render

O `render.yaml` cria um serviço web Python no plano gratuito (o processo usa cerca de 170 MB, dentro dos 512 MB). O modelo (`models/triage_model.pkl`) e o vocabulário vão no repositório, porque o treino precisa do MIMIC e não roda no Render.

1. No Render, **New > Blueprint**, escolha este repositório e a branch `develop`.
2. Aplique o Blueprint. A URL pública (`https://triai-ml-xxxx.onrender.com`) é o valor de `TRIAGE_MODEL_URL` da API Java.
3. Confira: `GET <url>/sintomas` deve listar as palavras-chave.

Se retreinar, commite o novo `.pkl` e o `vocabulary.json` juntos e mantenha as versões do `requirements.txt`: o modelo só carrega com o mesmo scikit-learn.
