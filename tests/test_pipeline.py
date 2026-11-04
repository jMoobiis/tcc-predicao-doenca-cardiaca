"""Testes básicos: dados, métricas, ausência de vazamento e determinismo."""
import inspect

import numpy as np
import pytest

from cardio_ml import limiares
from cardio_ml.avaliacao import diferencas_pareadas, indices_bootstrap, metricas
from cardio_ml.config import carregar_config
from cardio_ml.dados import carregar, dividir
from cardio_ml.preprocessamento import decidir_atributos, montar_preprocessador


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def dados(cfg):
    return carregar(cfg)


def test_dados_tem_920_registros_e_prevalencia_conhecida(dados):
    assert len(dados) == 920
    assert int(dados["alvo"].sum()) == 509
    assert dados.groupby("base").size().to_dict() == {
        "Cleveland": 303, "Hungria": 294, "Suica": 123, "Long Beach VA": 200}


def test_zeros_de_colesterol_viram_ausentes(dados):
    assert (dados["chol"] == 0).sum() == 0
    assert dados.loc[dados.base == "Suica", "chol"].isna().all()


def test_divisao_e_deterministica_e_estratificada(dados, cfg):
    a, b = dividir(dados, cfg), dividir(dados, cfg)
    assert list(a[1].index) == list(b[1].index)
    assert len(a[1]) == 184 and int(a[3].sum()) == 102


def test_descarte_usa_so_o_treino(dados, cfg):
    X_tr, X_te, *_ = dividir(dados, cfg)
    usados, descartados, aus = decidir_atributos(X_tr, cfg.limiar_descarte)
    assert descartados == ["ca", "thal"]
    assert aus["ca"] == pytest.approx(X_tr["ca"].isna().mean())


def test_preprocessador_aprende_so_com_o_treino(dados, cfg):
    X_tr, X_te, *_ = dividir(dados, cfg)
    usados, _, _ = decidir_atributos(X_tr, cfg.limiar_descarte)
    prep = montar_preprocessador(usados).fit(X_tr[usados])
    numericas = prep.transformers_[0][2]
    mediana_chol = prep.named_transformers_["num"].named_steps["imp"].statistics_[numericas.index("chol")]
    assert mediana_chol == pytest.approx(X_tr["chol"].median())


def test_metricas_em_matriz_conhecida():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.6, 0.1])
    r = metricas(y, p)
    assert (r["vp"], r["fn"], r["fp"], r["vn"]) == (2, 1, 1, 1)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(1 / 2)
    assert r["accuracy"] == pytest.approx(3 / 5)


def test_limiar_nao_recebe_dados_de_teste():
    # a escolha do limiar só aceita rótulos e probabilidades do treino
    assert list(inspect.signature(limiares.escolher_limiar).parameters) == ["y_treino", "prob_oof", "meta"]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    t = limiares.escolher_limiar(y, p, 0.9)
    assert metricas(y, p, t)["recall"] >= 0.9


def test_bootstrap_e_deterministico_e_tem_duas_classes():
    y = np.array([0] * 10 + [1] * 10)
    a, b = indices_bootstrap(y, 50, 42), indices_bootstrap(y, 50, 42)
    assert all(np.array_equal(i, j) for i, j in zip(a, b))
    assert all(len(set(y[i])) == 2 for i in a)


def test_diferenca_pareada_de_modelo_com_ele_mesmo_e_zero():
    amostras = {"A": {"roc_auc": [0.8, 0.9]}, "B": {"roc_auc": [0.8, 0.9]}}
    teste = {"A": {"roc_auc": 0.85}, "B": {"roc_auc": 0.85}}
    d = diferencas_pareadas(amostras, teste, ["A", "B"])[0]
    assert d["diferenca"] == 0 and d["ic"] == [0, 0] and not d["exclui_zero"]
