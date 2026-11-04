"""Orquestração: ambiente, análise exploratória, experimento principal e complementares.

As chaves já existentes no JSON (ambiente, exploratoria, preprocessamento, modelos,
cenario_ca_thal, hospital_fora) seguem exatamente o formato do script original
(tcc_heart_disease.py), o que permite conferir valor a valor que a reorganização não
mudou nenhum número. As análises acrescentadas na revisão ficam na chave "revisao".
"""
from __future__ import annotations

import json
import os
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.stats import skew

from . import figuras, limiares
from .avaliacao import (METRICAS, baselines, bootstrap, diferenca, diferencas_pareadas,
                        importancia_permutacao, indices_bootstrap, metricas)
from .config import Config
from .dados import BASES, CATEGORICAS, NUMERICAS, PREDITORES, carregar, dividir, baixar_dados
from .modelos import busca, catalogo, descrever_hiperparametros, limpar
from .preprocessamento import decidir_atributos
from .relatorios import Relatorio
from .treino import ajustar_final, validacao_aninhada


def fmt(m, r):
    return f"{r[m][0]:.4f} ±{r[m][1]:.3f}"


# ------------------------------------------------------------------ ambiente
def _memoria_gb():
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            estado = MEMORYSTATUSEX()
            estado.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
            return round(estado.ullTotalPhys / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def _processador():
    if sys.platform == "win32":
        try:
            import winreg
            chave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(chave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def ambiente(rel: Relatorio, cfg: Config):
    rel.titulo("AMBIENTE DE EXECUCAO (Quadro 6)")
    versoes = {
        "Python": platform.python_version(), "pandas": pd.__version__, "NumPy": np.__version__,
        "scikit-learn": sklearn.__version__, "Matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__, "SciPy": scipy.__version__,
    }
    for nome, versao in versoes.items():
        rel.log(f"{nome:14s} {versao}")
    hardware = {"sistema": platform.platform(), "processador": _processador(),
                "nucleos_logicos": os.cpu_count(), "memoria_gb": _memoria_gb()}
    rel.log(f"Sistema        {hardware['sistema']}")
    rel.log(f"Processador    {hardware['processador']} ({hardware['nucleos_logicos']} nucleos logicos)")
    rel.log(f"Memoria (GB)   {hardware['memoria_gb']}")
    rel.log(f"Semente        {cfg.semente}")
    rel.log("")
    rel.resultados["ambiente"] = {"versoes": versoes, "hardware": hardware, "semente": cfg.semente}


# ------------------------------------------------------------------ 4.1 exploratória
def exploratoria(dados, rel: Relatorio, cfg: Config):
    rel.titulo("4.1  ANALISE EXPLORATORIA DE DADOS")
    rel.log(f"Total de registros: {len(dados)}")
    rel.log(f"Atributos preditores no arquivo: {len(PREDITORES)}")
    rel.log("")
    rel.log("--- Tabela 1: distribuicao do alvo por base ---")
    t1 = dados.groupby("base")["alvo"].agg(
        Registros="count",
        Classe_0=lambda s: int((s == 0).sum()),
        Classe_1=lambda s: int((s == 1).sum())).reindex(BASES)
    t1.loc["Total"] = [len(dados), int((dados.alvo == 0).sum()), int((dados.alvo == 1).sum())]
    t1["pct_classe_1"] = 100 * t1["Classe_1"] / t1["Registros"]
    rel.log(t1.round(1).to_string()); rel.log("")

    rel.log("--- Tabela 2: percentual de valores ausentes por atributo e base ---")
    t2 = dados.groupby("base")[PREDITORES].apply(lambda g: g.isna().mean() * 100).T[BASES]
    t2["Total"] = dados[PREDITORES].isna().mean() * 100
    rel.log(t2.round(1).to_string()); rel.log("")

    rel.log("--- Tabela 3: estatisticas descritivas dos atributos numericos ---")
    t3 = dados[NUMERICAS].describe().T[["mean", "std", "min", "50%", "max", "count"]]
    t3["assimetria"] = [skew(dados[c].dropna()) for c in NUMERICAS]
    rel.log(t3.round(2).to_string()); rel.log("")

    rel.log("--- Apoio a discussao: ausencia de colesterol x diagnostico ---")
    chol_aus = dados["chol"].isna()
    pct_aus = 100 * dados.loc[chol_aus, "alvo"].mean()
    pct_pres = 100 * dados.loc[~chol_aus, "alvo"].mean()
    rel.log(f"% de positivos com chol AUSENTE:  {pct_aus:.1f}  (n={int(chol_aus.sum())})")
    rel.log(f"% de positivos com chol PRESENTE: {pct_pres:.1f}  (n={int((~chol_aus).sum())})")
    rel.log(f"Registros com chol >= 400 mg/dL: {int((dados.chol >= 400).sum())}")
    rel.log("")
    rel.resultados["exploratoria"] = {
        "tabela1": {b: {"registros": int(r.Registros), "classe0": int(r.Classe_0),
                        "classe1": int(r.Classe_1), "pct1": float(r.pct_classe_1)}
                    for b, r in t1.iterrows()},
        "tabela2": {a: {c: float(v) for c, v in linha.items()} for a, linha in t2.iterrows()},
        "tabela3": {a: {"media": float(r["mean"]), "dp": float(r["std"]), "min": float(r["min"]),
                        "mediana": float(r["50%"]), "max": float(r["max"]), "n": int(r["count"]),
                        "assimetria": float(r["assimetria"])}
                    for a, r in t3.iterrows()},
        "chol_ausente": {"pct_pos_ausente": float(pct_aus), "n_ausente": int(chol_aus.sum()),
                         "pct_pos_presente": float(pct_pres), "n_presente": int((~chol_aus).sum()),
                         "n_acima_400": int((dados.chol >= 400).sum())},
    }
    figuras.exploratorias(dados, cfg.pasta_saida)
    rel.log("Figuras exploratorias salvas em ./saida/"); rel.log("")


# ------------------------------------------------------------------ experimento principal
def principal(dados, rel: Relatorio, cfg: Config):
    X_tr, X_te, y_tr, y_te, b_tr, b_te = dividir(dados, cfg)
    usados, descartados, ausencia = decidir_atributos(X_tr, cfg.limiar_descarte)
    rel.titulo("PRE-PROCESSAMENTO (Secao 3.3)")
    rel.log(f"Treino: {len(X_tr)} registros | Teste: {len(X_te)} registros")
    rel.log(f"Limiar de descarte: {cfg.limiar_descarte:.0%} de ausencia, medido SO no treino")
    for c in PREDITORES:
        if ausencia[c] > 0:
            marca = "DESCARTADO" if c in descartados else ""
            rel.log(f"   {c:9s} {ausencia[c] * 100:5.1f}% ausente no treino  {marca}")
    rel.log(f"Atributos UTILIZADOS ({len(usados)}): {usados}"); rel.log("")
    rel.resultados["preprocessamento"] = {
        "n_treino": len(X_tr), "n_teste": len(X_te), "usados": usados, "descartados": descartados,
        "ausencia_treino": {c: float(ausencia[c] * 100) for c in PREDITORES},
        "teste_por_base": {b: int((b_te == b).sum()) for b in BASES},
    }

    nomes = list(catalogo(cfg.semente))
    cv, finais, escolhas, oofs = {}, {}, {}, {}
    for nome in nomes:
        rel.progresso(f"validacao aninhada: {nome}")
        cv[nome], escolhas[nome], oofs[nome] = validacao_aninhada(nome, usados, X_tr, y_tr, cfg)
        finais[nome] = ajustar_final(nome, usados, X_tr, y_tr, cfg)

    rel.titulo("4.2  DESEMPENHO DOS MODELOS")
    rel.log("--- Tabela 4: hiperparametros selecionados (busca em grade no treino inteiro) ---")
    for nome, gs in finais.items():
        rel.log(f"{nome}: {limpar(gs.best_params_)} | AUC media (5 dobras internas) = {gs.best_score_:.4f}")
    rel.log("")
    rel.log("Estabilidade da escolha: configuracao vencedora em cada uma das 10 dobras externas")
    for nome, lista in escolhas.items():
        contagem = pd.Series([json.dumps(e, sort_keys=True, default=str) for e in lista]).value_counts()
        rel.log(f"  {nome}: {len(contagem)} configuracoes distintas; mais frequente ({contagem.iloc[0]}x): "
                f"{contagem.index[0]}")
    rel.log("")
    rel.log("--- Tabela 5: validacao cruzada ANINHADA, 10 dobras externas (media ± desvio) ---")
    rel.log(f"{'Modelo':22s}" + "".join(f"{m:>16s}" for m in METRICAS))
    for nome, r in cv.items():
        rel.log(f"{nome:22s}" + "".join(f"{fmt(m, r):>16s}" for m in METRICAS))
    rel.log("")
    melhor = max(cv, key=lambda n: cv[n]["roc_auc"][0])
    rel.log(f"Melhor modelo pela AUC da validacao aninhada: {melhor}"); rel.log("")

    # ---------------- teste (uma única vez)
    probs = {n: gs.predict_proba(X_te)[:, 1] for n, gs in finais.items()}
    teste = {n: metricas(y_te, p) for n, p in probs.items()}
    rel.progresso("bootstrap do teste")
    indices = indices_bootstrap(y_te, cfg.n_bootstrap, cfg.semente)
    ic, amostras = bootstrap(y_te, probs, indices)

    rel.log("--- Tabela 6: desempenho no conjunto de teste [IC 95% bootstrap] ---")
    for nome, r in teste.items():
        partes = [f"{m}={r[m]:.4f} [{ic[nome][m][0]:.3f}; {ic[nome][m][1]:.3f}]" for m in METRICAS]
        rel.log(f"{nome}:")
        rel.log("    " + "  ".join(partes[:3]))
        rel.log("    " + "  ".join(partes[3:]))
    rel.log("")
    rel.log(f"Diferenca de AUC no teste: {melhor} menos cada outro modelo [IC 95% bootstrap pareado]")
    diferencas = {}
    for nome in nomes:
        if nome == melhor:
            continue
        diferencas[nome] = diferenca(amostras, teste, melhor, nome)
        rel.log(f"  vs {nome:20s} {diferencas[nome][0]:+.4f}  [{diferencas[nome][1]:+.3f}; {diferencas[nome][2]:+.3f}]")
    rel.log("")

    r = teste[melhor]
    rel.log(f"--- Tabela 7: matriz de confusao do melhor modelo ({melhor}), limiar 0,5 ---")
    rel.log(f"VN = {r['vn']}   FP = {r['fp']}")
    rel.log(f"FN = {r['fn']}   VP = {r['vp']}"); rel.log("")

    rel.log(f"--- Apoio a discussao da Tabela 7: limiar escolhido no TREINO ({melhor}) ---")
    rel.log("O limiar e o maior valor cuja revocacao, nas previsoes fora da dobra da validacao")
    rel.log("aninhada, atinge a meta. So depois ele e aplicado ao teste.")
    lim_melhor = {}
    for meta in cfg.metas_revocacao:
        t = limiares.escolher_limiar(y_tr, oofs[melhor], meta)
        rt = metricas(y_te, probs[melhor], t)
        lim_melhor[str(meta)] = {"limiar": float(t), **{k: float(v) for k, v in rt.items()}}
        rel.log(f"  meta de revocacao {meta:.0%}: limiar={t:.3f} -> no teste: revocacao={rt['recall']:.3f} "
                f"precisao={rt['precision']:.3f} especificidade={rt['specificity']:.3f} "
                f"FN={rt['fn']} FP={rt['fp']}")
    rel.log("")

    rel.log("--- Tabela 9: desempenho no teste por base hospitalar (limiar 0,5) ---")
    por_base = {}
    for nome in nomes:
        por_base[nome] = {}
        for b in BASES:
            m = (b_te == b).to_numpy()
            rb = metricas(y_te[m], probs[nome][m])
            rb.update(n=int(m.sum()), positivos=int(y_te[m].sum()))
            por_base[nome][b] = rb
    for b in BASES:
        rb = por_base[melhor][b]
        auc = "indefinida" if np.isnan(rb["roc_auc"]) else f"{rb['roc_auc']:.3f}"
        rel.log(f"  {b:14s} n={rb['n']:3d} positivos={rb['positivos']:3d}  acuracia={rb['accuracy']:.3f} "
                f"revocacao={rb['recall']:.3f} especificidade={rb['specificity']:.3f} AUC={auc}")
    rel.log("  (modelo: " + melhor + "; demais modelos no JSON)"); rel.log("")

    rel.progresso("importancia por permutacao")
    rel.log("--- Tabela 8: importancia por permutacao no teste (queda de AUC), Random Forest ---")
    imp_rf = importancia_permutacao(finais["Random Forest"].best_estimator_, X_te[usados], y_te,
                                    cfg.n_permutacoes, cfg.semente)
    rel.log(imp_rf.round(4).to_string()); rel.log("")
    imp_melhor = imp_rf
    if melhor != "Random Forest":
        rel.log(f"--- Importancia por permutacao no teste, {melhor} ---")
        imp_melhor = importancia_permutacao(finais[melhor].best_estimator_, X_te[usados], y_te,
                                            cfg.n_permutacoes, cfg.semente)
        rel.log(imp_melhor.round(4).to_string()); rel.log("")
    rel.log("--- Comparacao: importancia por reducao de impureza (MDI) do Random Forest, por atributo ---")
    rf = finais["Random Forest"].best_estimator_
    nomes_col = rf.named_steps["prep"].get_feature_names_out()
    mdi = pd.Series(rf.named_steps["clf"].feature_importances_, index=nomes_col)
    original = {n: n.split("__", 1)[1] for n in nomes_col}
    for n in nomes_col:
        for c in CATEGORICAS:
            if original[n].startswith(c + "_"):
                original[n] = c
    mdi_attr = mdi.groupby(original).sum().sort_values(ascending=False)
    rel.log(mdi_attr.round(4).to_string()); rel.log("")

    rel.resultados["modelos"] = {
        "melhor": melhor,
        "tabela4": {n: {"params": limpar(gs.best_params_), "auc_interna": float(gs.best_score_)}
                    for n, gs in finais.items()},
        "estabilidade": {n: [json.dumps(e, sort_keys=True, default=str) for e in l]
                         for n, l in escolhas.items()},
        "tabela5": cv,
        "tabela6": {n: {m: float(r[m]) for m in METRICAS} for n, r in teste.items()},
        "tabela6_ic": ic,
        "tabela7": {k: teste[melhor][k] for k in ("vn", "fp", "fn", "vp")},
        "diferencas_auc": diferencas,
        "limiares": lim_melhor,
        "por_base": {n: {b: {k: (None if isinstance(v, float) and np.isnan(v) else float(v))
                             for k, v in d.items()} for b, d in pb.items()}
                     for n, pb in por_base.items()},
        "tabela8_rf": {a: [float(v.media), float(v.dp)] for a, v in imp_rf.iterrows()},
        "importancia_melhor": {a: [float(v.media), float(v.dp)] for a, v in imp_melhor.iterrows()},
        "mdi_rf": {a: float(v) for a, v in mdi_attr.items()},
    }
    figuras.modelos(probs, y_te, teste, melhor, cfg.pasta_saida)

    revisao(rel, cfg, nomes, X_tr, y_tr, X_te, y_te, b_te, finais, oofs, probs, teste, amostras, indices)
    return X_tr, X_te, y_tr, y_te, usados


# ------------------------------------------------------------------ análises da revisão
def revisao(rel, cfg, nomes, X_tr, y_tr, X_te, y_te, b_te, finais, oofs, probs, teste, amostras, indices):
    rev = {}
    rel.titulo("REVISAO DO ORIENTADOR: ANALISES ACRESCENTADAS")

    # (1) todas as diferenças pareadas de AUC
    rel.log("--- (1) Diferencas pareadas de AUC no teste, todos os pares (a - b) [IC 95% bootstrap] ---")
    pares = diferencas_pareadas(amostras, teste, nomes, "roc_auc")
    for d in pares:
        marca = "EXCLUI ZERO" if d["exclui_zero"] else "inclui zero"
        rel.log(f"  {d['a']:20s} - {d['b']:20s} {d['diferenca']:+.4f} [{d['ic'][0]:+.3f}; {d['ic'][1]:+.3f}]  {marca}")
    rel.log("")
    rev["diferencas_pareadas_auc"] = pares
    rel.csv("diferencas_pareadas_auc.csv", ["modelo_a", "modelo_b", "diferenca_auc", "ic_inf", "ic_sup", "exclui_zero"],
            [[d["a"], d["b"], f"{d['diferenca']:.4f}", f"{d['ic'][0]:.4f}", f"{d['ic'][1]:.4f}", d["exclui_zero"]] for d in pares])

    # (2) limiares para todos os modelos
    rel.log("--- (2) Limiar escolhido nas predicoes fora da dobra do TREINO, aplicado ao teste ---")
    lim = {}
    linhas = []
    for nome in nomes:
        lim[nome] = limiares.analisar(y_tr, oofs[nome], y_te, probs[nome], cfg.metas_revocacao, cfg.limiar_padrao)
        for chave, r in lim[nome].items():
            rot = "padrao 0,5" if chave == "padrao" else f"meta {float(chave):.0%}"
            rel.log(f"  {nome:20s} {rot:11s} limiar={r['limiar']:.3f} revoc={r['recall']:.3f} "
                    f"espec={r['specificity']:.3f} VP={r['vp']:.0f} FN={r['fn']:.0f} FP={r['fp']:.0f} VN={r['vn']:.0f} "
                    f"dFN={r['delta_fn']:+d} dFP={r['delta_fp']:+d}")
            linhas.append([nome, rot, f"{r['limiar']:.3f}", f"{r['recall']:.3f}", f"{r['specificity']:.3f}",
                           int(r["vp"]), int(r["fn"]), int(r["fp"]), int(r["vn"]), r["delta_fn"], r["delta_fp"]])
    rel.log("")
    rev["limiares"] = lim
    rel.csv("limiares.csv", ["modelo", "limiar_regra", "limiar", "revocacao", "especificidade", "VP", "FN", "FP", "VN",
                             "delta_FN", "delta_FP"], linhas)

    # (3) matrizes de confusão de todos os modelos (limiar 0,5)
    rev["matrizes_confusao"] = {n: {k: int(teste[n][k]) for k in ("vn", "fp", "fn", "vp")} for n in nomes}

    # (4) Random Forest x SVM: matrizes e predições caso a caso
    rel.log("--- (4) Random Forest x SVM no teste (limiar 0,5) ---")
    pr = {n: (probs[n] >= cfg.limiar_padrao).astype(int) for n in ("Random Forest", "SVM")}
    yt = np.asarray(y_te)
    difer = np.where(pr["Random Forest"] != pr["SVM"])[0]
    casos = [{"indice_registro": int(X_te.index[i]), "base": str(b_te.iloc[i]), "real": int(yt[i]),
              "pred_rf": int(pr["Random Forest"][i]), "prob_rf": float(probs["Random Forest"][i]),
              "pred_svm": int(pr["SVM"][i]), "prob_svm": float(probs["SVM"][i])} for i in difer]
    rf_svm = {
        "matriz_rf": rev["matrizes_confusao"]["Random Forest"], "matriz_svm": rev["matrizes_confusao"]["SVM"],
        "matrizes_iguais": rev["matrizes_confusao"]["Random Forest"] == rev["matrizes_confusao"]["SVM"],
        "n_predicoes_diferentes": int(len(difer)), "casos_diferentes": casos,
        "probabilidades_identicas": bool(np.array_equal(probs["Random Forest"], probs["SVM"])),
        "correlacao_probabilidades": float(np.corrcoef(probs["Random Forest"], probs["SVM"])[0, 1]),
        "classes_dos_estimadores": [type(finais[n].best_estimator_.named_steps["clf"]).__name__
                                    for n in ("Random Forest", "SVM")],
        "objetos_distintos": finais["Random Forest"].best_estimator_ is not finais["SVM"].best_estimator_,
    }
    for n in ("Random Forest", "SVM"):
        m = rf_svm["matriz_rf" if n == "Random Forest" else "matriz_svm"]
        rel.log(f"  {n:14s} VN={m['vn']} FP={m['fp']} FN={m['fn']} VP={m['vp']}")
    rel.log(f"  predicoes diferentes: {len(difer)} de {len(yt)}; correlacao das probabilidades = "
            f"{rf_svm['correlacao_probabilidades']:.3f}")
    for c in casos:
        rel.log(f"    registro {c['indice_registro']} ({c['base']}), real={c['real']}: RF={c['pred_rf']} "
                f"(p={c['prob_rf']:.4f})  SVM={c['pred_svm']} (p={c['prob_svm']:.4f})")
    rel.log("")
    rev["rf_svm"] = rf_svm

    # (6) baselines ingênuos, com as MESMAS reamostras bootstrap
    rel.log("--- (6) Classificadores de referencia (DummyClassifier) no teste [IC 95%] ---")
    pb = baselines(X_tr, y_tr, X_te, cfg.semente)
    ic_b, _ = bootstrap(y_te, pb, indices)
    base = {}
    for n, p in pb.items():
        r = metricas(y_te, p)
        base[n] = {"metricas": {k: float(v) for k, v in r.items()}, "ic": ic_b[n]}
        partes = [f"{m}={r[m]:.4f} [{ic_b[n][m][0]:.3f}; {ic_b[n][m][1]:.3f}]" for m in METRICAS]
        rel.log(f"  {n}: " + "  ".join(partes))
    prev = float(np.mean(yt))
    rel.log(f"  prevalencia no teste: {prev:.4f} ({int(yt.sum())} de {len(yt)})")
    rel.log(f"  prevalencia no conjunto completo: {(np.sum(yt) + np.sum(y_tr)) / (len(yt) + len(y_tr)):.4f}")
    rel.log("")
    rev["baselines"] = base
    rev["prevalencia_teste"] = {"proporcao": prev, "positivos": int(yt.sum()), "n": int(len(yt))}

    # (5) hiperparâmetros: grade, escolhidos e efetivos
    rel.log("--- (5) Hiperparametros: grade testada, escolhidos e parametros efetivos ---")
    hp = descrever_hiperparametros(finais, cfg)
    for n, h in hp.items():
        rel.log(f"  {n} ({h['classe']}; {h['n_combinacoes']} combinacoes): escolhidos {h['escolhidos']}")
        rel.log(f"      efetivos: {h['parametros_efetivos']}")
    rel.log(f"  busca: GridSearchCV, {cfg.dobras_internas} dobras internas estratificadas, metrica {cfg.metrica_selecao}; "
            f"validacao aninhada com {cfg.dobras_externas} dobras externas; semente {cfg.semente}")
    rel.log("")
    rev["hiperparametros"] = {"metodo": "GridSearchCV", "dobras_internas": cfg.dobras_internas,
                              "dobras_externas": cfg.dobras_externas, "metrica": cfg.metrica_selecao,
                              "semente": cfg.semente, "modelos": hp}
    rel.resultados["revisao"] = rev


# ------------------------------------------------------------------ complementares
def cenario_ca_thal(X_tr, X_te, y_tr, y_te, usados, rel: Relatorio, cfg: Config):
    rel.titulo("EXPERIMENTO COMPLEMENTAR A (Tabela 10): cenario com ca e thal mantidos e imputados")
    extras = [c for c in ["ca", "thal"] if c not in usados]
    todos = [c for c in PREDITORES if c in usados or c in extras]
    rel.log(f"Atributos ({len(todos)}): {todos}")
    cenario = {}
    for nome in catalogo(cfg.semente):
        rel.progresso(f"cenario ca/thal: {nome}")
        resumo, _, _ = validacao_aninhada(nome, todos, X_tr[todos], y_tr, cfg)
        gs = ajustar_final(nome, todos, X_tr[todos], y_tr, cfg)
        r_te = metricas(y_te, gs.predict_proba(X_te[todos])[:, 1])
        cenario[nome] = {"auc_cv": resumo["roc_auc"], "auc_teste": float(r_te["roc_auc"]),
                         "recall_teste": float(r_te["recall"]),
                         "auc_cv_sem": rel.resultados["modelos"]["tabela5"][nome]["roc_auc"],
                         "auc_teste_sem": rel.resultados["modelos"]["tabela6"][nome]["roc_auc"]}
        if nome == "Random Forest":
            imp = importancia_permutacao(gs.best_estimator_, X_te[todos], y_te, cfg.n_permutacoes, cfg.semente)
            cenario[nome]["importancia"] = {a: float(v) for a, v in imp["media"].items()}
    rel.log(f"{'Modelo':22s}{'AUC CV sem':>16s}{'AUC CV com':>16s}{'AUC teste sem':>15s}{'AUC teste com':>15s}")
    for nome, c in cenario.items():
        rel.log(f"{nome:22s}{c['auc_cv_sem'][0]:>9.4f} ±{c['auc_cv_sem'][1]:.3f}"
                f"{c['auc_cv'][0]:>9.4f} ±{c['auc_cv'][1]:.3f}"
                f"{c['auc_teste_sem']:>15.4f}{c['auc_teste']:>15.4f}")
    rel.log("")
    rel.log("Importancia por permutacao (Random Forest, cenario com ca e thal):")
    for a, v in list(cenario["Random Forest"]["importancia"].items())[:6]:
        rel.log(f"   {a:9s} {v:.4f}")
    rel.log("")
    rel.resultados["cenario_ca_thal"] = cenario


def hospital_fora(dados, usados, rel: Relatorio, cfg: Config):
    from sklearn.metrics import roc_auc_score
    rel.titulo("EXPERIMENTO COMPLEMENTAR B (Tabela 11): validacao por hospital deixado de fora")
    rel.log("Atributos: os mesmos do experimento principal. Busca em grade com 5 dobras")
    rel.log("internas sobre as tres bases de treino; a quarta base e usada so para teste.")
    rng = np.random.default_rng(cfg.semente)
    loho = {}
    for b in BASES:
        treino, teste = dados[dados.base != b], dados[dados.base == b]
        loho[b] = {"n": len(teste), "positivos": int(teste.alvo.sum())}
        for nome in catalogo(cfg.semente):
            rel.progresso(f"hospital fora ({b}): {nome}")
            gs = busca(nome, usados, cfg).fit(treino[usados], treino.alvo)
            prob = gs.predict_proba(teste[usados])[:, 1]
            r = metricas(teste.alvo, prob)
            y = teste.alvo.to_numpy()
            aucs = []
            while len(aucs) < cfg.n_bootstrap:
                idx = rng.integers(0, len(y), len(y))
                if len(set(y[idx])) == 2:
                    aucs.append(roc_auc_score(y[idx], prob[idx]))
            loho[b][nome] = {k: float(r[k]) for k in METRICAS}
            loho[b][nome]["auc_ic"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    for b in BASES:
        rel.log(f"{b} (n={loho[b]['n']}, positivos={loho[b]['positivos']}):")
        for nome in catalogo(cfg.semente):
            r = loho[b][nome]
            rel.log(f"   {nome:22s} AUC={r['roc_auc']:.3f} [{r['auc_ic'][0]:.3f}; {r['auc_ic'][1]:.3f}] "
                    f"acuracia={r['accuracy']:.3f} revocacao={r['recall']:.3f} "
                    f"especificidade={r['specificity']:.3f}")
    rel.log("")
    rel.resultados["hospital_fora"] = loho


def executar(cfg: Config, completo: bool = True) -> Relatorio:
    """Reproduz TODOS os resultados do trabalho a partir dos dados brutos."""
    rel = Relatorio(cfg.pasta_saida)
    cfg.pasta_saida.mkdir(parents=True, exist_ok=True)
    baixar_dados(cfg)
    ambiente(rel, cfg)
    dados = carregar(cfg)
    exploratoria(dados, rel, cfg)
    X_tr, X_te, y_tr, y_te, usados = principal(dados, rel, cfg)
    if completo:
        cenario_ca_thal(X_tr, X_te, y_tr, y_te, usados, rel, cfg)
        hospital_fora(dados, usados, rel, cfg)
    figuras.diagrama_pipeline(cfg.pasta_saida)
    rel.gravar()
    return rel
