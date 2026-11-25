"""
Testes do ML-03: atributos de Doppler e entre antenas, cache, combinação de
conjuntos e seleção por validação.

Todos rodam sem o dataset (sinal sintético).

Uso:
    .venv/Scripts/python.exe -m pytest tests/test_ml03.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import doppler  # noqa: E402
from src.atributos import (  # noqa: E402
    CONJUNTOS_ATRIBUTOS,
    extrair_atributos,
    janelas_de_sequencia,
    montar_conjunto,
    validar_atributos,
)
from src.mmfi import AMBIENTES, SUJEITOS_TESTE_S2, ambiente_do_sujeito  # noqa: E402
from src.protocolos import (  # noqa: E402
    SUJEITOS,
    SUJEITOS_VALIDACAO_S2,
    dividir_s2,
    dividir_validacao_s2,
    listar_sequencias,
)


def _carregar_script(nome):
    especificacao = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / f"{nome}.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


baseline = _carregar_script("treinar_baseline")
selecao = _carregar_script("selecionar_atributos")

FORMA = (297, 3, 114, 10)
T = np.arange(297 * 10) / doppler.FS   # tempo da série contínua, em s


def _em_quadros(serie):
    """``(3, 114, 2970)`` -> ``(297, 3, 114, 10)`` (inverso da série contínua)."""
    return serie.reshape(3, 114, 297, 10).transpose(2, 0, 1, 3).astype(np.float32)


def _sequencia_aleatoria(seed=0):
    gerador = np.random.default_rng(seed)
    amp = (gerador.uniform(-10, 30, size=(1, 3, 114, 1))
           + gerador.normal(0, 2, size=FORMA)).astype(np.float32)
    fase = gerador.uniform(-np.pi, np.pi, size=FORMA).astype(np.float32)
    return amp, fase


# ==========================================================================
# Série contínua e referência do protótipo
# ==========================================================================

def test_recortes_seguem_a_serie_continua():
    gerador = np.random.default_rng(0)
    serie = gerador.normal(size=(3, 114, 2970))
    intervalos = janelas_de_sequencia(297, 30, 20)
    recortes = doppler._recortes(_em_quadros(serie), intervalos)
    for k, (i, f) in enumerate(intervalos):
        np.testing.assert_allclose(recortes[k], serie[:, :, 10 * i:10 * f], rtol=1e-6)


def _prototipo(amp, fase, intervalos):
    """Cálculo do protótipo descartável, janela por janela (referência).

    Única mudança: em ``antenas``, cos e sen médios no lugar do ângulo médio.
    """
    A = amp.transpose(1, 2, 0, 3).reshape(3, 114, -1).astype(np.float64)
    P = fase.transpose(1, 2, 0, 3).reshape(3, 114, -1).astype(np.float64)
    lin = 10 ** (A / 20)
    H = lin * np.exp(1j * P)
    ant, da, dc = [], [], []
    for i, j in intervalos:
        sl = slice(i * 10, j * 10)
        f = []
        for p, q in doppler.PARES:
            d = A[p, :, sl] - A[q, :, sl]
            ph = P[p, :, sl] - P[q, :, sl]
            c, sn = np.cos(ph).mean(-1), np.sin(ph).mean(-1)
            f += [d.mean(-1), d.std(-1), c, sn, np.hypot(c, sn)]
        ant.append(np.concatenate(f))
        x = lin[:, :, sl]
        x = x - x.mean(-1, keepdims=True)
        _, pa = welch(x, fs=100, nperseg=100, axis=-1)
        pa = pa.mean(1)
        pa = pa / (pa.sum(-1, keepdims=True) + 1e-12)
        da.append(np.log(pa + 1e-6).ravel())
        cm = []
        for p, q in doppler.PARES:
            z = H[p, :, sl] * np.conj(H[q, :, sl])
            z = z - z.mean(-1, keepdims=True)
            _, pc = welch(z, fs=100, nperseg=100, axis=-1, return_onesided=False)
            pc = pc.mean(0)
            cm.append(np.log(pc / (pc.sum() + 1e-12) + 1e-6))
        dc.append(np.concatenate(cm))
    return {k: np.array(v, np.float32) for k, v in
            (("antenas", ant), ("dop_amp", da), ("dop_cm", dc))}


def test_atributos_iguais_ao_prototipo():
    amp, fase = _sequencia_aleatoria(1)
    saida = doppler.atributos_sequencia(amp, fase)
    referencia = _prototipo(amp, fase, janelas_de_sequencia(297, 30))
    for nome, d in doppler.DIMENSOES.items():
        assert saida[nome].shape == (9, d) and saida[nome].dtype == np.float32
        np.testing.assert_allclose(saida[nome], referencia[nome], rtol=1e-5, atol=1e-5)
    assert doppler.DIMENSOES == {"antenas": 1710, "dop_amp": 153, "dop_cm": 300}


# ==========================================================================
# Matriz: senoide, rotação de fase, sinal constante
# ==========================================================================

@pytest.mark.parametrize("f", [3.0, 7.0, 21.0])
def test_senoide_na_amplitude_da_pico_no_bin_certo(f):
    linear = 5.0 + 2.0 * np.sin(2 * np.pi * f * T)
    serie = np.broadcast_to(20 * np.log10(linear), (3, 114, len(T)))
    amp = _em_quadros(serie)
    fase = np.zeros(FORMA, dtype=np.float32)
    dop = doppler.atributos_dop_amp(amp, janelas_de_sequencia(297, 30))
    freqs = doppler.frequencias_dop_amp()
    for antena in dop.reshape(9, 3, 51).reshape(-1, 51):
        assert freqs[np.argmax(antena)] == pytest.approx(f)
    assert np.all(np.isfinite(dop))


@pytest.mark.parametrize("f", [5.0, -5.0, 12.0, -30.0])
@pytest.mark.parametrize("par", [0, 1, 2])
def test_rotacao_de_fase_relativa_da_pico_com_sinal_certo(f, par):
    p, q = doppler.PARES[par]
    gerador = np.random.default_rng(0)
    serie_fase = np.broadcast_to(gerador.uniform(-np.pi, np.pi, size=(3, 114, 1)),
                                 (3, 114, len(T))).copy()
    serie_fase[p] += 2 * np.pi * f * T          # p − q gira a +f Hz
    fase = _em_quadros(np.angle(np.exp(1j * serie_fase)))  # crua, envolvida em ±π
    amp = np.full(FORMA, 10.0, dtype=np.float32)
    dop = doppler.atributos_dop_cm(amp, fase, janelas_de_sequencia(297, 30))
    freqs = doppler.frequencias_dop_cm()
    for janela in dop:
        espectro = janela.reshape(3, 100)[par]
        assert freqs[np.argmax(espectro)] == pytest.approx(f)


def test_sinal_constante_da_atributos_finitos():
    amp = np.full(FORMA, 12.5, dtype=np.float32)
    fase = np.full(FORMA, 0.3, dtype=np.float32)
    saida = doppler.atributos_sequencia(amp, fase)
    for nome, valores in saida.items():
        assert np.all(np.isfinite(valores)), nome
    antenas = saida["antenas"].reshape(9, 3, 5, 114)
    np.testing.assert_allclose(antenas[:, :, 1], 0, atol=1e-6)  # desvio da ΔdB
    np.testing.assert_allclose(antenas[:, :, 2], 1, atol=1e-6)  # cos(Δfase = 0)
    np.testing.assert_allclose(antenas[:, :, 3], 0, atol=1e-6)  # sen(Δfase = 0)
    np.testing.assert_allclose(antenas[:, :, 4], 1, atol=1e-6)  # fase relativa parada


def test_antenas_continuo_em_mais_ou_menos_pi():
    """Δfase logo acima de −π e logo abaixo de +π: quase o mesmo ângulo, mesmos atributos."""
    amp = np.full(FORMA, 10.0, dtype=np.float32)
    atributos = []
    for delta in (np.pi - 1e-3, -np.pi + 1e-3):
        fase = np.zeros(FORMA, dtype=np.float32)
        fase[:, 0] = delta
        atributos.append(doppler.atributos_antenas(amp, fase, janelas_de_sequencia(297, 30)))
    np.testing.assert_allclose(atributos[0], atributos[1], atol=1e-2)


def test_janela_curta_demais_para_o_doppler():
    amp, fase = _sequencia_aleatoria()
    with pytest.raises(ValueError, match="ao menos 10 quadros"):
        doppler.atributos_sequencia(amp, fase, janela=5)


# ==========================================================================
# Cache de atributos
# ==========================================================================

@pytest.fixture
def cache_falso(monkeypatch, tmp_path):
    sequencias = [("S01", "A01"), ("S02", "A01"), ("S03", "A02")]
    chamadas = []

    def carregar(sujeito, acao, raiz):
        chamadas.append((sujeito, acao))
        return _sequencia_aleatoria(SUJEITOS.index(sujeito) + 10 * int(acao[1:]))

    monkeypatch.setattr(doppler, "_conferir_dataset", lambda raiz, seqs: None)

    def construir(**kw):
        return doppler.construir_cache_atributos(
            tmp_path / "cache", tmp_path, sequencias=sequencias, carregar=carregar, **kw)

    return construir, chamadas, sequencias, tmp_path / "cache"


def test_cache_grava_e_abre(cache_falso):
    construir, chamadas, sequencias, destino = cache_falso
    construir()
    assert chamadas == sequencias
    cache = doppler.abrir_cache_atributos(destino)
    assert (cache.janela, cache.passo) == (30, 30)
    assert cache.linhas == {seq: i for i, seq in enumerate(sequencias)}
    esperado = doppler.atributos_sequencia(*_sequencia_aleatoria(SUJEITOS.index("S02") + 10))
    for nome, d in doppler.DIMENSOES.items():
        assert cache.conjuntos[nome].shape == (3, 9, d)
        np.testing.assert_array_equal(cache.conjuntos[nome][1], esperado[nome])


def test_cache_reaproveitado_com_mesma_janela_e_versao(cache_falso, capsys):
    construir, chamadas, _, _ = cache_falso
    construir()
    capsys.readouterr()
    chamadas.clear()
    construir()
    assert chamadas == []
    assert "AVISO" not in capsys.readouterr().out


@pytest.mark.parametrize("mudanca", ["janela", "passo", "versao"])
def test_cache_reconstruido_com_aviso(cache_falso, capsys, mudanca):
    construir, chamadas, sequencias, destino = cache_falso
    construir()
    chamadas.clear()
    capsys.readouterr()
    if mudanca == "versao":
        indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
        indice["versao"] = 0
        (destino / "indice.json").write_text(json.dumps(indice), encoding="utf-8")
        construir()
    elif mudanca == "janela":
        construir(janela=20)
    else:
        construir(passo=15)
    assert chamadas == sequencias
    assert "desatualizado" in capsys.readouterr().out
    cache = doppler.abrir_cache_atributos(destino)
    n = len(janelas_de_sequencia(297, cache.janela, cache.passo))
    assert cache.conjuntos["dop_cm"].shape == (3, n, 300)


def test_cache_sem_marcador_reconstruido(cache_falso, capsys):
    construir, chamadas, sequencias, destino = cache_falso
    construir()
    (destino / doppler.MARCADOR).unlink()
    with pytest.raises(FileNotFoundError):
        doppler.abrir_cache_atributos(destino)
    chamadas.clear()
    capsys.readouterr()
    construir()
    assert chamadas == sequencias
    assert "incompleto" in capsys.readouterr().out


# ==========================================================================
# Combinação de conjuntos em montar_conjunto
# ==========================================================================

def _cache_csi_sintetico(sequencias):
    pares = [_sequencia_aleatoria(40 + k) for k in range(len(sequencias))]
    amp = np.stack([a for a, _ in pares])
    fase = np.stack([f for _, f in pares])
    linhas = {seq: i for i, seq in enumerate(sequencias)}
    return amp, fase, linhas


def _cache_atributos_sintetico(sequencias, janela=30, passo=30):
    n = len(janelas_de_sequencia(297, janela, passo))
    gerador = np.random.default_rng(7)
    # Linhas em ordem diferente da do cache do CSI, para pegar troca de índice.
    ordem = list(reversed(sequencias))
    conjuntos = {nome: gerador.normal(size=(len(ordem), n, d)).astype(np.float32)
                 for nome, d in doppler.DIMENSOES.items()}
    return doppler.CacheAtributos(conjuntos, {s: i for i, s in enumerate(ordem)},
                                  janela, passo)


SEQS = [("S01", "A01"), ("S02", "A02"), ("S03", "A01")]
ACOES = ("A01", "A02")


def test_padrao_identico_ao_ml02():
    amp, fase, linhas = _cache_csi_sintetico(SEQS)
    referencia = np.concatenate([
        extrair_atributos(np.stack([amp[k, i:f] for i, f in janelas_de_sequencia(297, 30)]),
                          np.stack([fase[k, i:f] for i, f in janelas_de_sequencia(297, 30)]))
        for k in range(len(SEQS))])
    padrao = montar_conjunto(amp, fase, linhas, SEQS, ACOES)
    explicito = montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos=("base",),
                                cache_atributos=_cache_atributos_sintetico(SEQS))
    for X, y, g in (padrao, explicito):
        np.testing.assert_array_equal(X, referencia)
        np.testing.assert_array_equal(y, padrao[1])
        np.testing.assert_array_equal(g, padrao[2])
    # Normalização (ML-02) continua igual no base.
    np.testing.assert_array_equal(
        montar_conjunto(amp, fase, linhas, SEQS, ACOES, normalizacao="zscore")[0],
        montar_conjunto(amp, fase, linhas, SEQS, ACOES, normalizacao="zscore",
                        atributos="base")[0])


def test_combinacao_concatena_na_ordem_pedida():
    amp, fase, linhas = _cache_csi_sintetico(SEQS)
    cache = _cache_atributos_sintetico(SEQS)
    base = montar_conjunto(amp, fase, linhas, SEQS, ACOES)[0]
    X, y, g = montar_conjunto(amp, fase, linhas, SEQS, ACOES,
                              atributos=("dop_cm", "base", "dop_amp"), cache_atributos=cache)
    assert X.shape == (27, 300 + 1368 + 153)
    for k, seq in enumerate(SEQS):
        linhas_k = slice(9 * k, 9 * k + 9)
        np.testing.assert_array_equal(X[linhas_k, :300],
                                      cache.conjuntos["dop_cm"][cache.linhas[seq]])
        np.testing.assert_array_equal(X[linhas_k, 300:1668], base[linhas_k])
        np.testing.assert_array_equal(X[linhas_k, 1668:],
                                      cache.conjuntos["dop_amp"][cache.linhas[seq]])
    np.testing.assert_array_equal(y, np.repeat([0, 1, 0], 9))
    np.testing.assert_array_equal(g, np.repeat([0, 1, 2], 9))


def test_normalizacao_nao_mexe_nos_conjuntos_de_doppler():
    amp, fase, linhas = _cache_csi_sintetico(SEQS)
    cache = _cache_atributos_sintetico(SEQS)
    sem = montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos="dop_amp,antenas",
                          cache_atributos=cache)[0]
    com = montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos="dop_amp,antenas",
                          cache_atributos=cache, normalizacao="zscore")[0]
    np.testing.assert_array_equal(sem, com)


def test_juntar_da_o_mesmo_que_montar_conjunto():
    amp, fase, linhas = _cache_csi_sintetico(SEQS)
    cache = _cache_atributos_sintetico(SEQS)
    blocos = {nome: montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos=(nome,),
                                    cache_atributos=cache)[0]
              for nome in CONJUNTOS_ATRIBUTOS}
    for combinacao in selecao.CANDIDATOS:
        np.testing.assert_array_equal(
            selecao.juntar(blocos, combinacao),
            montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos=combinacao,
                            cache_atributos=cache)[0])


@pytest.mark.parametrize("pedido", ["base,xyz", ["xyz"], [], "base,base"])
def test_conjunto_invalido_lista_os_validos(pedido):
    with pytest.raises(ValueError) as erro:
        validar_atributos(pedido)
    for nome in CONJUNTOS_ATRIBUTOS:
        assert nome in str(erro.value)


def test_argparse_rejeita_conjunto_invalido():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="dop_cm"):
        baseline.ler_atributos("base,xyz")
    assert baseline.ler_atributos("base, dop_amp,dop_cm") == ("base", "dop_amp", "dop_cm")


def test_conjunto_extra_sem_cache_ou_com_janela_diferente():
    amp, fase, linhas = _cache_csi_sintetico(SEQS)
    with pytest.raises(ValueError, match="cache_atributos"):
        montar_conjunto(amp, fase, linhas, SEQS, ACOES, atributos=("base", "dop_amp"))
    with pytest.raises(ValueError, match="janela/passo"):
        montar_conjunto(amp, fase, linhas, SEQS, ACOES, janela=20, atributos=("dop_amp",),
                        cache_atributos=_cache_atributos_sintetico(SEQS))


def test_sufixo():
    assert baseline.montar_sufixo("s2", "rf", "p3", 30, 30, 0) == "s2_rf_j30_p30_seed0"
    assert baseline.montar_sufixo("s2", "rf", "p3", 30, 30, 0, "nenhuma",
                                  ("base",)) == "s2_rf_j30_p30_seed0"
    assert (baseline.montar_sufixo("s2", "rf", "p3", 30, 30, 0, "nenhuma",
                                   ("base", "dop_amp", "dop_cm"))
            == "s2_rf_j30_p30_seed0_abase-dop_amp-dop_cm")
    assert (baseline.montar_sufixo("s1", "svm", "p1", 30, 15, 2, "zscore", "dop_cm")
            == "s1_svm_p1_j30_p15_seed2_nzscore_adop_cm")


# ==========================================================================
# Validação e seleção
# ==========================================================================

def test_validacao_disjunta_do_teste_e_do_treino():
    treino_s2, teste = dividir_s2()
    restante, validacao = dividir_validacao_s2()
    sujeitos_val = {s for s, _ in validacao}
    assert sujeitos_val == set(SUJEITOS_VALIDACAO_S2)
    assert not sujeitos_val & set(SUJEITOS_TESTE_S2)
    assert not sujeitos_val & {s for s, _ in restante}
    assert not {s for s, _ in restante} & set(SUJEITOS_TESTE_S2)
    assert sorted(restante + validacao) == sorted(treino_s2)
    assert not set(teste) & (set(restante) | set(validacao))
    assert sorted(ambiente_do_sujeito(s) for s in SUJEITOS_VALIDACAO_S2) == list(AMBIENTES)
    assert len(validacao) == 4 * 27 and len(restante) == 28 * 27


def test_conferir_validacao_recusa_sujeito_do_teste():
    treino, teste = dividir_s2(("A01",))
    with pytest.raises(ValueError, match=r"\['S05'\]"):
        selecao.conferir_validacao(treino, teste, ("S04", "S05"))
    # Sujeito de SUJEITOS_TESTE_S2 ausente do teste passado: ainda é citado.
    with pytest.raises(ValueError, match=r"\['S10'\]"):
        selecao.conferir_validacao(treino + [("S10", "A01")],
                                   [s for s in teste if s[0] != "S10"], ("S10",))
    with pytest.raises(ValueError, match="fora do treino"):
        selecao.conferir_validacao([s for s in treino if s[0] != "S04"], teste, ("S04",))


def _problema_sintetico(acoes=("A01", "A02", "A03"), janelas=2, seed=0):
    """``util`` separa as ações; ``ruido`` não. A validação deve preferir ``util``."""
    treino, teste = dividir_s2(acoes)
    gerador = np.random.default_rng(seed)
    centros = gerador.normal(size=(len(acoes), 4)) * 3

    def blocos(seqs):
        g = np.repeat(np.arange(len(seqs)), janelas)
        y = np.array([acoes.index(a) for _, a in seqs])[g]
        return ({"util": (centros[y] + gerador.normal(size=(len(g), 4))).astype(np.float32),
                 "ruido": gerador.normal(size=(len(g), 4)).astype(np.float32)}, y, g)

    return treino, teste, blocos(treino), blocos(teste)


def test_selecao_escolhe_pela_validacao_antes_do_teste(monkeypatch):
    monkeypatch.setattr("src.atributos.CONJUNTOS_ATRIBUTOS", ("util", "ruido"))
    treino, teste, (b_tr, y_tr, g_tr), (b_te, y_te, g_te) = _problema_sintetico()
    eventos = []
    original = selecao.treinar_e_medir

    def espiao(X_tr, y_tr_, X_te, y_te_, g_te_, n, seed, acertos=None):
        eventos.append(("teste" if len(X_te) == len(y_te) and np.array_equal(y_te_, y_te)
                        else "validacao", X_te.shape[1]))
        return original(X_tr, y_tr_, X_te, y_te_, g_te_, n, seed, acertos)

    monkeypatch.setattr(selecao, "treinar_e_medir", espiao)

    def ao_escolher(parcial):
        assert parcial["teste"] is None
        assert all(e[0] == "validacao" for e in eventos)
        eventos.append(("escolha", parcial["escolhida"]))

    resultado = selecao.selecionar(
        b_tr, y_tr, g_tr, treino, b_te, y_te, g_te, teste, 3,
        candidatos=[("ruido",), ("util",), ("util", "ruido")],
        seeds_validacao=(0, 1), seeds_teste=(0, 1, 2), referencia=("ruido",),
        ao_escolher=ao_escolher, log=lambda _: None)

    escolha = [i for i, e in enumerate(eventos) if e[0] == "escolha"]
    assert len(escolha) == 1
    assert all(e[0] == "validacao" for e in eventos[:escolha[0]])
    assert all(e[0] == "teste" for e in eventos[escolha[0] + 1:])
    assert resultado["escolhida"][0] == "util"
    assert set(resultado["teste"]) == {",".join(resultado["escolhida"]), "ruido"}
    for item in resultado["teste"].values():
        assert len(item["execucoes"]) == 3
    # Validação: 4 sujeitos × 3 ações, 2 janelas cada.
    assert resultado["validacao"][0]["execucoes"][0]["por_janela"]["n"] == 4 * 3 * 2
    assert resultado["n_sequencias"]["validacao"] == 4 * 3
    util = resultado["teste"][",".join(resultado["escolhida"])]["resumo"]
    assert util["por_sequencia"]["media"] > resultado["teste"]["ruido"]["resumo"]["por_sequencia"]["media"]
    # Comparação pareada escolhida − referência, por seed e na média.
    comparacao = resultado["comparacao_pareada"]
    assert [c["seed"] for c in comparacao["por_seed"]] == [0, 1, 2]
    for item in comparacao["por_seed"] + [comparacao["media_das_seeds"]]:
        for visao in ("por_janela", "por_sequencia"):
            d = item[visao]
            assert d["ic95"][0] <= d["diferenca"] <= d["ic95"][1]
            assert 0.0 <= d["fracao_menor_igual_zero"] <= 1.0
    escolhida = resultado["teste"][",".join(resultado["escolhida"])]
    for seed, item in enumerate(comparacao["por_seed"]):
        for visao in ("por_janela", "por_sequencia"):
            assert item[visao]["diferenca"] == pytest.approx(
                escolhida["execucoes"][seed][visao]["acuracia"]
                - resultado["teste"]["ruido"]["execucoes"][seed][visao]["acuracia"])

    # Mesmos argumentos e seeds -> mesma escolha e mesmos números.
    de_novo = selecao.selecionar(
        b_tr, y_tr, g_tr, treino, b_te, y_te, g_te, teste, 3,
        candidatos=[("ruido",), ("util",), ("util", "ruido")],
        seeds_validacao=(0, 1), seeds_teste=(0, 1, 2), referencia=("ruido",),
        log=lambda _: None)
    for chave in ("escolhida", "validacao", "teste"):
        assert de_novo[chave] == resultado[chave]


def test_selecao_nunca_treina_com_sujeitos_de_teste_ou_validacao(monkeypatch):
    monkeypatch.setattr("src.atributos.CONJUNTOS_ATRIBUTOS", ("util", "ruido"))
    treino, teste, (b_tr, y_tr, g_tr), (b_te, y_te, g_te) = _problema_sintetico(("A01",))
    # Marca cada linha com o índice do sujeito, para conferir quem entra em cada ajuste.
    sujeito_tr = np.array([SUJEITOS.index(s) for s, _ in treino])[g_tr]
    sujeito_te = np.array([SUJEITOS.index(s) for s, _ in teste])[g_te]
    b_tr = {"util": sujeito_tr[:, None].astype(np.float32)}
    b_te = {"util": sujeito_te[:, None].astype(np.float32)}
    vistos = []

    def espiao(X_tr, y_tr_, X_te, y_te_, g_te_, n, seed, acertos=None):
        vistos.append(({SUJEITOS[int(v)] for v in X_tr[:, 0]},
                       {SUJEITOS[int(v)] for v in X_te[:, 0]}))
        return {"seed": seed, **{v: {"n": 1, "acuracia": 0.5, "ic95": [0, 1], "f1_macro": 0}
                                 for v in ("por_janela", "por_sequencia")}}

    monkeypatch.setattr(selecao, "treinar_e_medir", espiao)
    selecao.selecionar(b_tr, y_tr, g_tr, treino, b_te, y_te, g_te, teste, 1,
                       candidatos=[("util",)], seeds_validacao=(0,), seeds_teste=(0,),
                       referencia=("util",), log=lambda _: None)
    (fit_val, avaliado_val), (fit_teste, avaliado_teste) = vistos
    assert avaliado_val == set(SUJEITOS_VALIDACAO_S2)
    assert not fit_val & (set(SUJEITOS_VALIDACAO_S2) | set(SUJEITOS_TESTE_S2))
    assert not fit_teste & set(SUJEITOS_TESTE_S2)
    assert fit_teste == {s for s, _ in treino}
    assert avaliado_teste == set(SUJEITOS_TESTE_S2)


def test_escolher_desempata_pela_janela_e_depois_pela_ordem():
    def item(nome, seq, jan):
        return {"atributos": [nome], "resumo": {"por_sequencia": {"media": seq},
                                                "por_janela": {"media": jan}}}
    assert selecao.escolher([item("a", 0.2, 0.1), item("b", 0.3, 0.1)]) == ("b",)
    assert selecao.escolher([item("a", 0.3, 0.1), item("b", 0.3, 0.2)]) == ("b",)
    assert selecao.escolher([item("a", 0.3, 0.2), item("b", 0.3, 0.2)]) == ("a",)


def test_bootstrap_pareado():
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=float)
    b = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    r = selecao.bootstrap_pareado(a, b, seed=0)
    assert r["diferenca"] == pytest.approx(3 / 8)
    assert r["ic95"][0] <= r["diferenca"] <= r["ic95"][1]
    assert r["ic95"][0] >= 0 and r["fracao_menor_igual_zero"] > 0  # a ≥ b sempre
    igual = selecao.bootstrap_pareado(a, a, seed=0)
    assert igual["diferenca"] == 0 and igual["ic95"] == [0.0, 0.0]
    assert igual["fracao_menor_igual_zero"] == 1.0
    # Por janela, agrupado por sequência: o mesmo sorteio para os dois lados.
    grupos = np.repeat(np.arange(4), 2)
    r = selecao.bootstrap_pareado(a, b, grupos, seed=0)
    assert r["diferenca"] == pytest.approx(3 / 8)
    with pytest.raises(ValueError):
        selecao.bootstrap_pareado(a, b[:-1])
