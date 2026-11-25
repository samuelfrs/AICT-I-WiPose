"""
Testes da CDF por ação: gravação das probabilidades por janela, CDF empírica e
a figura ``resultado_4_cdf_por_acao.png``.

Rodam sem o dataset (probabilidades sintéticas).

Uso:
    .venv/Scripts/python.exe -m pytest tests/test_cdf.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.avaliacao import (  # noqa: E402
    avaliar,
    carregar_probabilidades,
    cdf_empirica,
    probabilidade_da_correta,
    salvar_probabilidades,
)
from src.mmfi import PROTOCOLO_3  # noqa: E402

ACOES = tuple(sorted(PROTOCOLO_3))


def _carregar_script(nome):
    especificacao = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / f"{nome}.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def _proba_sintetica(n_pessoas=2, janelas=3, forca=2.0, seed=0):
    """Probabilidades por janela de ``n_pessoas`` × 27 ações, ``janelas`` cada.

    A ação ``A01`` recebe muito mais massa na classe certa que a ``A27``, para
    a ordem dos painéis ser previsível.
    """
    gerador = np.random.default_rng(seed)
    sequencias = [(f"S{p + 1:02d}", a) for p in range(n_pessoas) for a in ACOES]
    grupos = np.repeat(np.arange(len(sequencias)), janelas)
    y = np.array([ACOES.index(sequencias[g][1]) for g in grupos])
    logits = gerador.normal(size=(len(y), len(ACOES)))
    bonus = forca * (len(ACOES) - y) / len(ACOES)      # A01 mais fácil, A27 mais difícil
    logits[np.arange(len(y)), y] += bonus
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, y, grupos, sequencias


def test_salvar_e_carregar_probabilidades_ida_e_volta(tmp_path):
    proba, y, grupos, sequencias = _proba_sintetica()
    caminho = salvar_probabilidades(tmp_path / "sub" / "p.npz", proba, y, grupos, ACOES, sequencias)
    lido = carregar_probabilidades(caminho)
    np.testing.assert_allclose(lido["probabilidades"], proba, rtol=1e-6)
    np.testing.assert_array_equal(lido["y"], y)
    np.testing.assert_array_equal(lido["grupos"], grupos)
    assert list(lido["acoes"]) == list(ACOES)
    assert [tuple(s) for s in lido["sequencias"]] == sequencias
    # Refaz a métrica gravada no JSON a partir do arquivo.
    original = avaliar(proba, y, grupos, len(ACOES))
    refeita = avaliar(lido["probabilidades"].astype(np.float64), lido["y"], lido["grupos"], len(ACOES))
    assert refeita["por_janela"]["acuracia"] == original["por_janela"]["acuracia"]
    assert refeita["por_sequencia"]["acuracia"] == original["por_sequencia"]["acuracia"]


@pytest.mark.parametrize("estrago, mensagem", [
    ("janelas", "mesmo número de janelas"),
    ("colunas", "Uma coluna"),
    ("grupos", "fora do intervalo"),
])
def test_salvar_probabilidades_recusa_entrada_inconsistente(tmp_path, estrago, mensagem):
    proba, y, grupos, sequencias = _proba_sintetica()
    if estrago == "janelas":
        y = y[:-1]
    elif estrago == "colunas":
        proba = proba[:, :-1]
    else:
        grupos = grupos.copy()
        grupos[0] = len(sequencias)
    with pytest.raises(ValueError, match=mensagem):
        salvar_probabilidades(tmp_path / "p.npz", proba, y, grupos, ACOES, sequencias)


def test_probabilidade_da_correta_pega_a_coluna_da_acao_verdadeira():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(probabilidade_da_correta(proba, np.array([0, 1])), [0.7, 0.3])


def test_cdf_empirica_ordena_e_vai_ate_um():
    x, f = cdf_empirica(np.array([0.5, 0.1, 0.3, 0.3]))
    np.testing.assert_allclose(x, [0.1, 0.3, 0.3, 0.5])
    np.testing.assert_allclose(f, [0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        cdf_empirica(np.array([]))


def _gravar_fontes(pasta, figuras_mod):
    """Grava dois modelos sintéticos com os nomes que a figura procura."""
    for (nome, arquivo, _), forca, seed in zip(figuras_mod.FONTES_CDF, (6.0, 2.0), (0, 1)):
        proba, y, grupos, sequencias = _proba_sintetica(forca=forca, seed=seed)
        salvar_probabilidades(pasta / arquivo, proba, y, grupos, ACOES, sequencias)


def test_resumo_por_acao_e_figura(tmp_path, monkeypatch):
    figuras = _carregar_script("gerar_figuras_resultados")
    monkeypatch.setattr(figuras, "FIGURAS", tmp_path / "fig")
    (tmp_path / "fig").mkdir()
    _gravar_fontes(tmp_path, figuras)

    resumo = figuras.resumo_cdf_por_acao(pasta=tmp_path)
    assert resumo["acoes"] == list(ACOES)
    assert [m["nome"] for m in resumo["modelos"]] == [f[0] for f in figuras.FONTES_CDF]
    forte, fraco = resumo["modelos"]
    assert forte["acuracia_janela"] > fraco["acuracia_janela"]
    a01 = forte["por_acao"]["A01"]
    assert a01["n"] == 6                                    # 2 pessoas × 3 janelas
    assert forte["por_acao"]["A01"]["mediana"] > forte["por_acao"]["A27"]["mediana"]
    assert 0.0 <= a01["acima_do_acaso"] <= 1.0

    destino = figuras.figura_cdf_por_acao(pasta=tmp_path)
    assert destino == tmp_path / "fig" / "resultado_4_cdf_por_acao.png"
    assert destino.stat().st_size > 10_000
    tabela = json.loads((tmp_path / "resumo_cdf_por_acao.json").read_text(encoding="utf-8"))
    assert set(tabela["modelos"][0]["por_acao"]) == set(ACOES)
    assert "p_correta" not in tabela["modelos"][0]["por_acao"]["A01"]
    # A tabela vem na ordem dos painéis: da ação mais fácil para a mais difícil.
    medianas = [v["mediana"] for v in tabela["modelos"][0]["por_acao"].values()]
    assert medianas == sorted(medianas, reverse=True)


def test_figura_com_um_modelo_so_e_sem_nenhum(tmp_path, monkeypatch, capsys):
    figuras = _carregar_script("gerar_figuras_resultados")
    monkeypatch.setattr(figuras, "FIGURAS", tmp_path / "fig")
    (tmp_path / "fig").mkdir()
    with pytest.raises(FileNotFoundError):
        figuras.resumo_cdf_por_acao(pasta=tmp_path)

    nome, arquivo, _ = figuras.FONTES_CDF[0]
    proba, y, grupos, sequencias = _proba_sintetica()
    salvar_probabilidades(tmp_path / arquivo, proba, y, grupos, ACOES, sequencias)
    resumo = figuras.resumo_cdf_por_acao(pasta=tmp_path)
    assert [m["nome"] for m in resumo["modelos"]] == [nome]
    assert "fica de fora" in capsys.readouterr().out
    assert figuras.figura_cdf_por_acao(pasta=tmp_path).is_file()


def test_modelos_com_acoes_diferentes_falham(tmp_path):
    figuras = _carregar_script("gerar_figuras_resultados")
    (_, arq1, _), (_, arq2, _) = figuras.FONTES_CDF
    proba, y, grupos, sequencias = _proba_sintetica()
    salvar_probabilidades(tmp_path / arq1, proba, y, grupos, ACOES, sequencias)
    invertidas = tuple(reversed(ACOES))
    salvar_probabilidades(tmp_path / arq2, proba, y, grupos, invertidas, sequencias)
    with pytest.raises(ValueError, match="ações diferentes"):
        figuras.resumo_cdf_por_acao(pasta=tmp_path)
