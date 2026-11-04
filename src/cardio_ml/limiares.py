"""Escolha de limiar de decisão por meta de revocação, usando SÓ o conjunto de treino."""
from __future__ import annotations

import numpy as np

from .avaliacao import metricas


def escolher_limiar(y_treino, prob_oof, meta: float) -> float:
    """Maior limiar cuja revocação nas predições fora da dobra (treino) atinge a meta.

    A função recebe apenas rótulos e probabilidades do TREINO: o teste não participa.
    """
    y = np.asarray(y_treino)
    for t in np.unique(prob_oof)[::-1]:
        if metricas(y, prob_oof, t)["recall"] >= meta:
            return float(t)
    raise ValueError("nenhum limiar atinge a meta")


def analisar(y_treino, prob_oof, y_te, prob_te, metas, limiar_padrao: float) -> dict:
    """Para cada meta: limiar escolhido no treino e desempenho no teste, com a variação
    de falsos negativos e falsos positivos em relação ao limiar padrão."""
    base = metricas(y_te, prob_te, limiar_padrao)
    saida = {"padrao": {"limiar": limiar_padrao, **{k: float(v) for k, v in base.items()},
                        "delta_fn": 0, "delta_fp": 0}}
    for meta in metas:
        t = escolher_limiar(y_treino, prob_oof, meta)
        r = metricas(y_te, prob_te, t)
        revoc_treino = metricas(np.asarray(y_treino), prob_oof, t)["recall"]
        saida[str(meta)] = {"limiar": t, "revocacao_oof_treino": float(revoc_treino),
                            **{k: float(v) for k, v in r.items()},
                            "delta_fn": int(r["fn"] - base["fn"]),
                            "delta_fp": int(r["fp"] - base["fp"])}
    return saida
