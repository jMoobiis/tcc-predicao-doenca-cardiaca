# Predição de doença cardíaca com aprendizado de máquina: pipeline reprodutível

Artefato de software do Trabalho de Conclusão de Curso de **João Victor Mobiglia Podenciano**
(Análise e Desenvolvimento de Sistemas, Fatec Ribeirão Preto, 2026).

O pipeline compara cinco classificadores supervisionados (Regressão Logística, Árvore de Decisão,
Random Forest, SVM e k-NN) nas **quatro bases** do conjunto UCI Heart Disease (920 registros) e
reproduz, a partir dos dados brutos e com um único comando, **todas** as tabelas, números e
figuras do artigo e da monografia.

## Instalação

Requer Python 3.11 ou superior (os resultados oficiais foram obtidos com Python 3.14.4).

```bash
git clone https://github.com/<usuario>/tcc-predicao-doenca-cardiaca.git
cd tcc-predicao-doenca-cardiaca
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Execução

```bash
python -m cardio_ml run            # pipeline completo (cerca de 10 minutos em 16 núcleos)
python -m cardio_ml run --rapido   # só o experimento principal, sem os complementares
python -m pytest                   # testes automatizados
```

A saída vai para `saida/`:

| Arquivo | Conteúdo |
|---|---|
| `resultados.txt` | todos os números, legíveis, na ordem das tabelas do trabalho |
| `resultados.json` | os mesmos números em formato estruturado |
| `tabelas/*.csv` | diferenças pareadas de AUC e análise de limiares |
| `fig_*.png` | figuras da análise exploratória, curvas ROC, matriz de confusão e diagrama do pipeline |

### Conferência de reprodutibilidade

```bash
python -m cardio_ml verificar referencia/resultados.json
```

Compara valor a valor a execução atual com a de referência (`referencia/resultados.json`, gerada
pelo script original do trabalho). Ambiente e tempo de execução são ignorados. Na versão publicada,
os **935 valores** da referência são reproduzidos com **0 diferenças**.

## Arquitetura

```
config.toml                  semente, divisão, dobras, bootstrap, metas de revocação
src/cardio_ml/
├── __main__.py              linha de comando (run, verificar)
├── config.py                leitura do config.toml
├── dados.py                 download/carga das 4 bases, alvo binário, divisão 80/20
├── preprocessamento.py      descarte por ausência (só no treino), imputação, one-hot, padronização
├── modelos.py               os 5 classificadores e as grades de hiperparâmetros
├── treino.py                validação cruzada aninhada (10 externas × 5 internas) e ajuste final
├── avaliacao.py             métricas, IC por bootstrap, diferenças pareadas, baselines
├── limiares.py              limiar por meta de revocação, escolhido só no treino
├── experimentos.py          orquestração: principal, cenário com ca/thal, hospital deixado de fora
├── relatorios.py            resultados.txt/.json, CSV e comparação entre execuções
└── figuras.py               figuras e diagrama do pipeline
tests/                       testes automatizados (pytest)
dados/                       os 4 arquivos processed.*.data do UCI
```

Fluxo de dados: `dados` → divisão estratificada → `preprocessamento` (ajustado dentro de cada
dobra, encadeado ao classificador em um `Pipeline` do scikit-learn, o que impede vazamento) →
`treino` (busca em grade aninhada, seleção pela AUC) → `avaliacao`/`limiares` (teste usado uma
única vez) → `relatorios`/`figuras`.

Garantias de reprodutibilidade: semente única (42) em todas as etapas aleatórias; versões das
bibliotecas fixadas em `requirements.txt`; mesmas reamostras bootstrap para todos os modelos;
o limiar de decisão é escolhido nas predições fora da dobra do treino, nunca no teste.

## Dados e licença

Base: JANOSI, A.; STEINBRUNN, W.; PFISTERER, M.; DETRANO, R. **Heart Disease** [Dataset].
UCI Machine Learning Repository, 1989. DOI: <https://doi.org/10.24432/C52P4X>.

Os arquivos de `dados/` são redistribuídos sem modificação sob a licença
**Creative Commons Attribution 4.0 International (CC BY 4.0)**, conforme indicado na página do
conjunto (<https://archive.ics.uci.edu/dataset/45/heart+disease>). Se os arquivos não estiverem
presentes, o pipeline os baixa do repositório UCI.

O código deste repositório é distribuído sob a licença MIT (ver `LICENSE`).

## Aviso

Trabalho acadêmico e exploratório. Nenhum modelo aqui produzido se destina a uso clínico.
