"""Carga das quatro bases do UCI Heart Disease (id 45) e divisão treino/teste."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config

ARQUIVOS = {
    "Cleveland":     "processed.cleveland.data",
    "Hungria":       "processed.hungarian.data",
    "Suica":         "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}
BASES = list(ARQUIVOS)
COLUNAS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
PREDITORES = COLUNAS[:-1]
NUMERICAS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAS = ["cp", "restecg", "slope", "thal"]
BINARIAS = ["sex", "fbs", "exang"]

# nome em português (forma completa, forma curta) de cada variável da base
NOMES_PT = {
    "age":      ("Idade", "Idade"),
    "sex":      ("Sexo", "Sexo"),
    "cp":       ("Tipo de dor torácica", "Dor torácica"),
    "trestbps": ("Pressão arterial em repouso", "Pressão em repouso"),
    "chol":     ("Colesterol sérico", "Colesterol"),
    "fbs":      ("Glicemia de jejum > 120 mg/dL", "Glicemia de jejum"),
    "restecg":  ("Eletrocardiograma em repouso", "ECG em repouso"),
    "thalach":  ("Frequência cardíaca máxima atingida", "FC máxima"),
    "exang":    ("Angina induzida por exercício", "Angina no exercício"),
    "oldpeak":  ("Depressão do segmento ST induzida por exercício", "Depressão de ST"),
    "slope":    ("Inclinação do segmento ST no pico do exercício", "Inclinação de ST"),
    "ca":       ("Número de vasos principais coloridos por fluoroscopia", "Vasos na fluoroscopia"),
    "thal":     ("Resultado da cintilografia com tálio", "Cintilografia com tálio"),
    "num":      ("Diagnóstico angiográfico", "Diagnóstico"),
    "alvo":     ("Diagnóstico (alvo binário)", "Diagnóstico"),
}


def baixar_dados(cfg: Config) -> None:
    """Baixa os arquivos do UCI apenas se ainda não estiverem em `dados/`."""
    cfg.pasta_dados.mkdir(parents=True, exist_ok=True)
    for arq in ARQUIVOS.values():
        destino = cfg.pasta_dados / arq
        if not destino.exists():
            print(f"Baixando {arq} ...")
            urllib.request.urlretrieve(cfg.url_base + arq, destino)


def carregar(cfg: Config) -> pd.DataFrame:
    quadros = []
    for nome, arq in ARQUIVOS.items():
        df = pd.read_csv(cfg.pasta_dados / arq, names=COLUNAS, na_values="?", header=None)
        df["base"] = nome
        quadros.append(df)
    dados = pd.concat(quadros, ignore_index=True)
    for col in COLUNAS:                       # colunas lidas como texto por causa do '?'
        dados[col] = pd.to_numeric(dados[col], errors="coerce")
    for col in ["chol", "trestbps"]:          # zeros fisiologicamente impossíveis = ausência
        dados.loc[dados[col] == 0, col] = np.nan
    # alvo binário: 0 = ausência, 1..4 = presença
    dados["alvo"] = (dados["num"] > 0).astype(int)
    return dados


def dividir(dados: pd.DataFrame, cfg: Config):
    """Divisão estratificada treino/teste; devolve também a base de origem de cada registro."""
    return train_test_split(dados[PREDITORES], dados["alvo"], dados["base"],
                            test_size=cfg.proporcao_teste, stratify=dados["alvo"],
                            random_state=cfg.semente)
