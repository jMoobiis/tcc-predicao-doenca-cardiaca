"""Pré-processamento: descarte por ausência (decidido no treino) e transformações em pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dados import BINARIAS, CATEGORICAS, NUMERICAS, PREDITORES


def decidir_atributos(X_tr, limiar_descarte: float):
    """Atributos com ausência acima do limiar, medida SÓ no treino, são descartados."""
    ausencia = X_tr[PREDITORES].isna().mean()
    descartados = [c for c in PREDITORES if ausencia[c] > limiar_descarte]
    usados = [c for c in PREDITORES if c not in descartados]
    return usados, descartados, ausencia


def montar_preprocessador(colunas_usadas) -> ColumnTransformer:
    num = [c for c in NUMERICAS if c in colunas_usadas]
    binarias = [c for c in BINARIAS if c in colunas_usadas]
    cat = [c for c in CATEGORICAS if c in colunas_usadas]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), num),
        # binárias já estão em 0/1: só imputação pela moda, sem padronizar
        ("bin", SimpleImputer(strategy="most_frequent"), binarias),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
