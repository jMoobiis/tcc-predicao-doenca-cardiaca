"""Métricas, intervalos de confiança por bootstrap, diferenças pareadas e baselines."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score

METRICAS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def metricas(y, prob, limiar: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(prob) >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = {
        "accuracy": (vp + vn) / len(y),
        "precision": vp / (vp + fp) if vp + fp else float("nan"),
        "recall": vp / (vp + fn) if vp + fn else float("nan"),
        "specificity": vn / (vn + fp) if vn + fp else float("nan"),
        "roc_auc": roc_auc_score(y, prob) if len(set(y)) == 2 else float("nan"),
        "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp),
    }
    p, s = r["precision"], r["recall"]
    r["f1"] = 2 * p * s / (p + s) if p + s else float("nan")
    return r


def indices_bootstrap(y_te, n: int, semente: int) -> list[np.ndarray]:
    """Reamostragens do teste com as duas classes presentes; as MESMAS para todos os modelos."""
    rng = np.random.default_rng(semente)
    y = np.asarray(y_te)
    indices = []
    while len(indices) < n:
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            indices.append(idx)
    return indices


def bootstrap(y_te, probs: dict, indices) -> tuple[dict, dict]:
    """IC de 95% (percentil) de cada métrica para cada modelo."""
    y = np.asarray(y_te)
    amostras = {n: {m: [] for m in METRICAS} for n in probs}
    for idx in indices:
        for n, p in probs.items():
            r = metricas(y[idx], p[idx])
            for m in METRICAS:
                amostras[n][m].append(r[m])
    ic = {n: {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
              for m, v in d.items()} for n, d in amostras.items()}
    return ic, amostras


def diferenca(amostras, teste, a: str, b: str, metrica: str = "roc_auc") -> list[float]:
    """[pontual, IC inferior, IC superior] de (a - b), pareada nas mesmas reamostras."""
    d = np.array(amostras[a][metrica]) - np.array(amostras[b][metrica])
    pontual = teste[a][metrica] - teste[b][metrica]
    return [float(pontual), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def diferencas_pareadas(amostras, teste, nomes, metrica: str = "roc_auc") -> list[dict]:
    """Todas as combinações de pares (a, b) na ordem dos modelos."""
    linhas = []
    for a, b in combinations(nomes, 2):
        pontual, lo, hi = diferenca(amostras, teste, a, b, metrica)
        linhas.append({"a": a, "b": b, "metrica": metrica, "diferenca": pontual,
                       "ic": [lo, hi], "exclui_zero": bool(lo > 0 or hi < 0)})
    return linhas


def baselines(X_tr, y_tr, X_te, semente: int) -> dict:
    """Classificadores de referência ingênuos (DummyClassifier)."""
    saida = {}
    for rotulo, estrategia in (("Classe majoritária", "most_frequent"),
                               ("Estratificado", "stratified")):
        d = DummyClassifier(strategy=estrategia, random_state=semente).fit(X_tr, y_tr)
        saida[rotulo] = d.predict_proba(X_te)[:, 1].astype(float)
    return saida


def importancia_permutacao(modelo, X_te, y_te, n_repeticoes: int, semente: int) -> pd.DataFrame:
    r = permutation_importance(modelo, X_te, y_te, scoring="roc_auc",
                               n_repeats=n_repeticoes, random_state=semente, n_jobs=-1)
    return pd.DataFrame({"media": r.importances_mean, "dp": r.importances_std},
                        index=X_te.columns).sort_values("media", ascending=False)
