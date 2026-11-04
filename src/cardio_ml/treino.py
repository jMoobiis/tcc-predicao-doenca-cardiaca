"""Validação cruzada aninhada e ajuste final dos modelos."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .avaliacao import METRICAS, metricas
from .config import Config
from .modelos import busca, limpar


def validacao_aninhada(nome: str, usados, X_tr, y_tr, cfg: Config):
    """10 dobras externas; em cada uma, busca em grade com 5 dobras internas.

    Devolve o resumo (média, desvio) por métrica, a configuração vencedora em cada dobra
    e as probabilidades fora da dobra (out-of-fold) de todo o treino, usadas na escolha
    do limiar sem tocar no teste.
    """
    externa = StratifiedKFold(n_splits=cfg.dobras_externas, shuffle=True, random_state=cfg.semente)
    por_dobra, escolhas = [], []
    oof = np.zeros(len(y_tr))
    for i_tr, i_va in externa.split(X_tr, y_tr):
        gs = busca(nome, usados, cfg).fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr])
        prob = gs.predict_proba(X_tr.iloc[i_va])[:, 1]
        oof[i_va] = prob
        por_dobra.append(metricas(y_tr.iloc[i_va], prob))
        escolhas.append(limpar(gs.best_params_))
    resumo = {m: (float(np.mean([d[m] for d in por_dobra])),
                  float(np.std([d[m] for d in por_dobra])))
              for m in METRICAS}
    return resumo, escolhas, oof


def ajustar_final(nome: str, usados, X_tr, y_tr, cfg: Config):
    """Busca em grade sobre o treino inteiro: configuração final avaliada no teste."""
    return busca(nome, usados, cfg).fit(X_tr, y_tr)
