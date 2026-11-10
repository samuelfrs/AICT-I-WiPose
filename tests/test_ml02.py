"""
Testes do ML-02: normalização por sequência e diagnóstico de identificação do sujeito.

Todos rodam sem o dataset (sinal sintético).

Uso:
    .venv/Scripts/python.exe -m pytest tests/test_ml02.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.atributos import (  # noqa: E402
    MODOS_NORMALIZACAO,
    extrair_atributos,
    janelas_de_sequencia,
    montar_conjunto,
    normalizar_sequencia,
)
from src.protocolos import SUJEITOS, dividir_s2, listar_sequencias  # noqa: E402


def _carregar_script():
    especificacao = importlib.util.spec_from_file_location(
        "treinar_baseline", RAIZ / "scripts" / "treinar_baseline.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


baseline = _carregar_script()


def _sequencia(seed=0, forma=(297, 3, 114, 10)):
    """Sinal com média e escala diferentes por antena × subportadora."""
    gerador = np.random.default_rng(seed)
    nivel = gerador.uniform(-20, 20, size=(1, forma[1], forma[2], 1))
    escala = gerador.uniform(0.1, 5, size=(1, forma[1], forma[2], 1))
    return (nivel + escala * gerador.normal(size=forma)).astype(np.float32)


# ==========================================================================
# normalizar_sequencia
# ==========================================================================

def test_nenhuma_devolve_a_entrada_sem_alteracao():
    x = _sequencia()
    copia = x.copy()
    saida = normalizar_sequencia(x, "nenhuma")
    assert saida is x
    np.testing.assert_array_equal(x, copia)


def test_media_zera_a_media_por_antena_e_subportadora():
    x = _sequencia(1)
    saida = normalizar_sequencia(x, "media")
    assert saida.shape == x.shape and saida.dtype == np.float32
    np.testing.assert_allclose(saida.mean(axis=(0, 3), dtype=np.float64), 0, atol=1e-5)
    # O desvio não muda: só a média foi removida.
    np.testing.assert_allclose(saida.std(axis=(0, 3)), x.std(axis=(0, 3)), rtol=1e-4)


def test_zscore_media_zero_e_desvio_um():
    x = _sequencia(2)
    saida = normalizar_sequencia(x, "zscore")
    np.testing.assert_allclose(saida.mean(axis=(0, 3), dtype=np.float64), 0, atol=1e-5)
    np.testing.assert_allclose(saida.std(axis=(0, 3), dtype=np.float64), 1, rtol=1e-5)


def test_zscore_bate_com_a_formula_do_prototipo():
    x = _sequencia(3).astype(np.float64)
    esperado = (x - x.mean(axis=(0, 3), keepdims=True)) / x.std(axis=(0, 3), keepdims=True)
    np.testing.assert_allclose(normalizar_sequencia(x, "zscore"), esperado,
                               rtol=1e-5, atol=1e-5)


def test_normalizacao_independe_entre_antenas_e_subportadoras():
    x = _sequencia(4)
    alterado = x.copy()
    alterado[:, 1, 50] = alterado[:, 1, 50] * 7 + 100  # mexe só numa célula
    a, b = normalizar_sequencia(x, "zscore"), normalizar_sequencia(alterado, "zscore")
    outras = np.ones((3, 114), dtype=bool)
    outras[1, 50] = False
    np.testing.assert_array_equal(a[:, outras], b[:, outras])
    np.testing.assert_allclose(a[:, 1, 50], b[:, 1, 50], atol=1e-5)  # z-score é afim-invariante


def test_zscore_sinal_constante_vira_zero_sem_nan():
    x = _sequencia(5)
    x[:, 0, 3] = 0.1          # constante "difícil" de representar em binário
    x[:, 2, 100] = -37.25
    saida = normalizar_sequencia(x, "zscore")
    assert np.all(np.isfinite(saida))
    assert np.all(saida[:, 0, 3] == 0) and np.all(saida[:, 2, 100] == 0)
    # As outras subportadoras continuam normalizadas.
    np.testing.assert_allclose(saida[:, 1].std(axis=(0, 2), dtype=np.float64), 1, rtol=1e-5)


def test_modo_invalido_lista_os_modos_validos():
    with pytest.raises(ValueError) as erro:
        normalizar_sequencia(_sequencia(), "minmax")
    for modo in MODOS_NORMALIZACAO:
        assert modo in str(erro.value)


def test_montar_conjunto_rejeita_modo_invalido():
    amp = np.zeros((1, 297, 3, 114, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="nenhuma"):
        montar_conjunto(amp, amp, {("S01", "A01"): 0}, [("S01", "A01")], ("A01",),
                        normalizacao="minmax")


# ==========================================================================
# montar_conjunto
# ==========================================================================

def _cache_sintetico(n=3):
    amp = np.stack([_sequencia(10 + i) for i in range(n)])
    fase = np.stack([_sequencia(20 + i) for i in range(n)])
    sequencias = [(f"S{i + 1:02d}", "A01") for i in range(n)]
    linhas = {seq: i for i, seq in enumerate(sequencias)}
    return amp, fase, linhas, sequencias


def test_montar_conjunto_nenhuma_identico_ao_ml01():
    amp, fase, linhas, sequencias = _cache_sintetico()
    padrao = montar_conjunto(amp, fase, linhas, sequencias, ("A01",))
    explicito = montar_conjunto(amp, fase, linhas, sequencias, ("A01",),
                                normalizacao="nenhuma")
    # Referência: o cálculo do ML-01, à mão, sem normalização.
    intervalos = janelas_de_sequencia(297, 30)
    referencia = np.concatenate([
        extrair_atributos(np.stack([amp[k, i:f] for i, f in intervalos]),
                          np.stack([fase[k, i:f] for i, f in intervalos]))
        for k in range(len(sequencias))
    ])
    for X, y, g in (padrao, explicito):
        np.testing.assert_array_equal(X, referencia)
        np.testing.assert_array_equal(y, padrao[1])
        np.testing.assert_array_equal(g, padrao[2])


def test_montar_conjunto_normaliza_a_sequencia_inteira_antes_das_janelas():
    amp, fase, linhas, sequencias = _cache_sintetico(2)
    X, _, _ = montar_conjunto(amp, fase, linhas, sequencias, ("A01",),
                              normalizacao="zscore")
    intervalos = janelas_de_sequencia(297, 30)
    a = normalizar_sequencia(amp[1], "zscore")
    f = normalizar_sequencia(fase[1], "zscore")
    esperado = extrair_atributos(np.stack([a[i:j] for i, j in intervalos]),
                                 np.stack([f[i:j] for i, j in intervalos]))
    np.testing.assert_allclose(X[9:], esperado, rtol=1e-6)
    # Estatística da sequência inteira, não da janela: a média de uma janela
    # isolada não é exatamente zero.
    assert np.abs(X[:9, :342]).max() > 1e-3


# ==========================================================================
# Script: sufixo e diagnóstico
# ==========================================================================

def test_sufixo_sem_normalizacao_igual_ao_ml01():
    assert baseline.montar_sufixo("s2", "rf", "p3", 30, 30, 0) == "s2_rf_j30_p30_seed0"
    assert (baseline.montar_sufixo("s1", "svm", "p1", 30, 15, 2, "nenhuma")
            == "s1_svm_p1_j30_p15_seed2")


@pytest.mark.parametrize("modo", ["media", "zscore"])
def test_sufixo_com_normalizacao(modo):
    assert (baseline.montar_sufixo("s2", "rf", "p3", 30, 30, 0, modo)
            == f"s2_rf_j30_p30_seed0_n{modo}")


def test_diagnostico_sujeito_detecta_assinatura():
    acoes = ("A01", "A02", "A03", "A04")
    sequencias = listar_sequencias(acoes)
    gerador = np.random.default_rng(0)
    assinatura = gerador.normal(size=(len(SUJEITOS), 8))
    grupos = np.repeat(np.arange(len(sequencias)), 3)
    sujeito = np.array([SUJEITOS.index(s) for s, _ in sequencias])[grupos]
    com_assinatura = assinatura[sujeito] + 0.05 * gerador.normal(size=(len(grupos), 8))
    sem_assinatura = gerador.normal(size=(len(grupos), 8))

    alto = baseline.diagnosticar_sujeito(com_assinatura, grupos, sequencias, acoes, seed=0)
    baixo = baseline.diagnosticar_sujeito(sem_assinatura, grupos, sequencias, acoes, seed=0)
    assert alto["acaso"] == pytest.approx(1 / 40)
    assert alto["acuracia"] > 0.9
    assert baixo["acuracia"] < 0.2
    assert alto["ic95"][0] <= alto["acuracia"] <= alto["ic95"][1]
    # S1 com 4 ações: 10 sequências de teste por ação, 3 janelas cada.
    assert alto["n_janelas_teste"] == 4 * 10 * 3


def test_diagnostico_sujeito_exige_todas_as_sequencias():
    acoes = ("A01",)
    sequencias = listar_sequencias(acoes)[:-1]
    X = np.zeros((len(sequencias), 2))
    with pytest.raises(ValueError, match="sem atributos"):
        baseline.diagnosticar_sujeito(X, np.arange(len(sequencias)), sequencias,
                                      acoes, seed=0)


def _dados_com_assinatura(acoes, janelas=3, seed=0):
    sequencias = listar_sequencias(acoes)
    gerador = np.random.default_rng(seed)
    assinatura = gerador.normal(size=(len(SUJEITOS), 8))
    grupos = np.repeat(np.arange(len(sequencias)), janelas)
    sujeito = np.array([SUJEITOS.index(s) for s, _ in sequencias])[grupos]
    X = assinatura[sujeito] + 0.5 * gerador.normal(size=(len(grupos), 8))
    return X, grupos, sequencias


def test_diagnostico_sujeito_independe_da_ordem_das_linhas():
    acoes = ("A01", "A02", "A03")
    X, grupos, sequencias = _dados_com_assinatura(acoes)
    referencia = baseline.diagnosticar_sujeito(X, grupos, sequencias, acoes, seed=0)

    # Mesmo dado em outra ordem: sequências embaralhadas e grupos renumerados.
    gerador = np.random.default_rng(1)
    permutacao = gerador.permutation(len(sequencias))  # nova posição -> antiga
    nova_posicao = np.argsort(permutacao)
    outras_seqs = [sequencias[g] for g in permutacao]
    # Linhas agrupadas na nova ordem das sequências; dentro de cada sequência,
    # as janelas mantêm a ordem original (como montar_conjunto produz).
    linhas = np.argsort(nova_posicao[grupos], kind="stable")
    assert not np.array_equal(linhas, np.arange(len(X)))
    outro = baseline.diagnosticar_sujeito(X[linhas], nova_posicao[grupos[linhas]],
                                          outras_seqs, acoes, seed=0)
    assert outro == referencia


def test_juntar_para_diagnostico_com_split_s2():
    acoes = ("A01", "A02")
    treino, teste = dividir_s2(acoes)
    janelas = 3

    def bloco(sequencias):
        grupos = np.repeat(np.arange(len(sequencias)), janelas)
        sujeitos = np.array([SUJEITOS.index(s) for s, _ in sequencias])[grupos]
        return sujeitos[:, None].astype(np.float32), grupos

    X_tr, g_tr = bloco(treino)
    X_te, g_te = bloco(teste)
    X, grupos, sequencias = baseline.juntar_para_diagnostico(
        X_tr, g_tr, treino, X_te, g_te, teste)
    assert len(X) == len(grupos) == (len(treino) + len(teste)) * janelas
    assert sorted(sequencias) == listar_sequencias(acoes)
    for linha, g in zip(X[:, 0], grupos):
        assert SUJEITOS[int(linha)] == sequencias[g][0]
