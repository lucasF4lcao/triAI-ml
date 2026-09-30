# triAI-ml

Serviço de classificação de triagem (Protocolo de Manchester) usado pelo `triAI-api-java`. Treinado com o MIMIC-IV-ED. É um serviço interno: quem fala FHIR com os hospitais é a API Java.

## Como funciona

O modelo prediz o **ESI (1 a 5)** registrado pelos enfermeiros do MIMIC-IV-ED a partir de 300 features:

- **desvio da faixa normal** de cada sinal vital (frequência cardíaca, respiratória, pressões, temperatura e saturação), em vez do valor cru. Faixas de adulto: FC 60-100, FR 12-20, PAS 100-140, PAD 60-90, temperatura 36-37,8 °C e SpO2 >= 95%. Dentro da faixa o desvio é zero;
- dor (0 a 10) e chegada de ambulância ou helicóptero;
- 287 palavras-chave de queixa principal (palavras e bigramas em inglês, do MIMIC), cada uma com rótulo em português.

As 5 classes são tratadas como uma **escala ordenada**: 4 classificadores `HistGradientBoosting` estimam a chance de o ESI ser menor ou igual a cada nível, e as probabilidades por classe saem da diferença entre eles. Nos desvios dos sinais vitais, na dor e na chegada de ambulância há **restrição monotônica**: piorar um sinal nunca deixa o paciente menos urgente. Por isso, variar os sinais **dentro da faixa normal não muda a resposta**.

A classe escolhida é o maior valor de `probabilidade x peso`. Os pesos (`[1, 1, 1, 2, 4]`) são ajustados em uma validação separada, para a distribuição prevista ficar perto da real, sem perder o recall do Vermelho e do Laranja (mais de 2 pontos) nem mais de 1 ponto de acurácia. A `confianca` devolvida é a probabilidade real da classe escolhida, sem o peso.

O ESI vira cor de Manchester por equivalência ordinal (1 Vermelho, 2 Laranja, 3 Amarelo, 4 Verde, 5 Azul), e a cor é forçada para **Vermelho** quando há sinais vitais críticos: SpO2 < 90, PAS < 80, FC fora de 40-150, FR fora de 8-30 ou temperatura fora de 35-40,5 °C. É uma adaptação metodológica, não uma equivalência clínica perfeita (concordância ESI x MTS de κ = 0,51 na literatura).

## Vocabulário de queixas

`shared/vocabulary.json` é a fonte única da lista de palavras-chave (código, termo em inglês, rótulo em português e `selectable`). O treino, a inferência e o `GET /sintomas` leem esse arquivo. Mudar a lista exige retreinar.

Só 163 dos 287 itens são **queixas completas** (`selectable: true`, como "Dor no dedo do pé"). Os outros 124 são fragmentos internos do modelo ("Dedo do pé", "Dor", "Inferior") e não devem ser oferecidos ao usuário. Escolher uma queixa completa ativa os fragmentos dela automaticamente, como acontece no texto de treino. Nenhum rótulo se repete entre os selecionáveis.

A queixa chega de três formas, combinadas: códigos em `sintomas`, texto livre em inglês e texto livre em português (casado pelo rótulo, sem acentos). Quando um rótulo contém outro ("Dor no dedo do pé" contém "Dor no dedo"), vale só o mais longo.

## Estrutura

```
app/        API FastAPI: fábrica (factory), schemas e inferência (classifier)
shared/     usado pelo treino e pela API: config, features, faixas clínicas, modelo ordinal, decisão, vocabulário e Manchester
training/   dataset (limpeza e features), tuning (pesos de decisão) e train (treino e avaliação)
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

O treino leva cerca de 15 minutos e usa cerca de 2 GB de memória no pico. Em máquinas com pouca RAM, rode com um limite de memória para não derrubar o sistema.

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
| `ambulancia` | booleano |

Campos desconhecidos, como `sexo_masculino` e `idade`, são aceitos e ignorados: o modelo não usa sexo nem idade.

O campo `selecionavel` de `/sintomas` indica se o item pode ser oferecido ao usuário.

Devolve `{"corPrevista": "Laranja", "confianca": 0.71}`. Campos desconhecidos são ignorados.

**`GET /sintomas`**: devolve `[{"codigo": "chest-pain", "termo": "chest pain", "rotulo": "Dor torácica"}, ...]`.

Atenção: sem nenhum sinal vital, o modelo tende a devolver Vermelho, porque no MIMIC a ausência de sinais costuma indicar paciente em estado crítico. A API Java recusa Bundles sem sinais vitais reconhecidos para não repassar esse resultado.

## Testes

```
python -m pytest
```

## Métricas

Teste de 20% dos dados (83.620 casos), que não participa do treino nem do ajuste dos pesos. Nenhuma etapa usa o alvo para preencher dados.

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| ESI 1 (Vermelho) | 0,76 | 0,63 | 0,69 |
| ESI 2 (Laranja) | 0,66 | 0,59 | 0,62 |
| ESI 3 (Amarelo) | 0,71 | 0,77 | 0,74 |
| ESI 4 (Verde) | 0,44 | 0,45 | 0,44 |
| ESI 5 (Azul) | 0,29 | 0,17 | 0,21 |

Acurácia de 68,0%, F1-macro de 0,54, e 98,8% dos casos saem a no máximo um nível de ESI do valor real. A distribuição prevista acompanha a real (Verde 7,1% previsto contra 6,8% real). Não rodei validação cruzada para este desenho.

Em relação ao modelo anterior (multiclasse, valores crus): a acurácia caiu 0,9 ponto, o F1-macro subiu de 0,48 para 0,54, o recall do Verde foi de 0,16 para 0,45 e o do Azul de 0,07 para 0,17. O modelo anterior mudava a classe em 7% dos pacientes com todos os sinais normais só por variar os sinais dentro da faixa normal.

## Limitações conhecidas

- **Subtriagem.** O modelo ainda tende ao Amarelo: 38% dos casos Laranja e 10% dos Vermelho do teste saíram como Amarelo ou menos urgente. A regra de sinais vitais críticos só protege os extremos.
- **Azul é raro nos dados** (0,26% dos casos) e o modelo o acerta pouco. Mesmo entre pacientes com sinais normais e queixa banal, o mais comum no MIMIC é Verde. Produzir Azul com confiança exigiria uma regra clínica validada por um profissional.
- **O sexo não entra no modelo, de propósito.** Com ele, o mesmo quadro clínico dava cores diferentes para homem e mulher (dor nas costas: Verde para o homem, Amarelo para a mulher). O efeito vinha da correlação nos dados, e não de um critério clínico.
- **Sinal vital ausente** continua sendo informação para o modelo, porque no MIMIC a ausência costuma indicar paciente crítico. Um Bundle sem nenhum sinal vital é recusado pela API Java.
- **Entradas incoerentes** (por exemplo, pressão sistólica 85 com diastólica 76) caem onde o modelo quase não tem dados e podem dar respostas menos confiáveis. Com valores coerentes, piorar um sinal sempre aumenta a urgência.
- **Dados americanos.** O ESI do MIMIC é rotulado por enfermeiros dos EUA, e as palavras-chave vêm de texto em inglês. O desempenho num hospital brasileiro não está medido. Queixas em português fora da lista só contam pelo casamento dos rótulos.
- **Sem idade.** O MIMIC-IV-ED não traz a idade (está em outra tabela, fora do acesso atual).
- As métricas medem o ESI. A cor devolvida pela API pode virar Vermelho pela regra de sinais vitais críticos.

## Deploy no Render

O `render.yaml` cria um serviço web Python no plano gratuito (o processo usa cerca de 170 MB, dentro dos 512 MB). O modelo (`models/triage_model.pkl`) e o vocabulário vão no repositório, porque o treino precisa do MIMIC e não roda no Render.

1. No Render, **New > Blueprint**, escolha este repositório e a branch `develop`.
2. Aplique o Blueprint. A URL pública (`https://triai-ml-xxxx.onrender.com`) é o valor de `TRIAGE_MODEL_URL` da API Java.
3. Confira: `GET <url>/sintomas` deve listar as palavras-chave.

Se retreinar, commite o novo `.pkl` e o `vocabulary.json` juntos e mantenha as versões do `requirements.txt`: o modelo só carrega com o mesmo scikit-learn.
