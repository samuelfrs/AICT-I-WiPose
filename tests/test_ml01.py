"""
Testes do ML-01: fase vetorizada, protocolos, janelas, atributos, métricas e cache.

Como em test_mmfi.py, a maior parte roda sem o dataset (sinal sintético). Os
marcados com ``@precisa_dataset`` constroem um cache pequeno a partir do MM-Fi
em disco e são pulados se ele não estiver lá.

Uso:
    .venv/Scripts/python.exe -m pytest tests/test_ml01.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache_csi  # noqa: E402
from src.atributos import (  # noqa: E402
    N_ATRIBUTOS,
    extrair_atributos,
    janelas_de_sequencia,
    montar_conjunto,
)
from src.avaliacao import agregar_por_sequencia, avaliar, ic_bootstrap  # noqa: E402
from src.mmfi import (  # noqa: E402
    PROTOCOLO_1,
    PROTOCOLO_3,
    SUJEITOS_TESTE_S2,
    carregar_csi_sequencia,
    formato_disponivel,
    sanitizar_fase,
    sanitizar_fase_lote,
)
from src.protocolos import (  # noqa: E402
    dividir_s1_agrupado,
    dividir_s2,
    listar_sequencias,
)

precisa_dataset = pytest.mark.skipif(
    formato_disponivel("S01", "A01") == "ausente",
    reason="MM-Fi não encontrado. Aponte a variável de ambiente MMFI_DATASET.",
)


# ==========================================================================
# Fase vetorizada
# ==========================================================================

def _fase_sintetica(forma, seed=0):
    """Rampa linear com salto no meio + ruído, envolvida em ±π."""
    gerador = np.random.default_rng(seed)
    k = np.arange(forma[-2])[:, None]
    rampa = 0.15 * k + 1.7 * (k >= 57)
    fase = rampa + gerador.normal(0, 0.3, size=forma)
    return np.angle(np.exp(1j * fase)).astype(np.float32)


def test_sanitizar_fase_lote_igual_a_versao_por_quadro():
    fase = _fase_sintetica((4, 3, 114, 10))
    lote = sanitizar_fase_lote(fase)
    referencia = np.stack([sanitizar_fase(q) for q in fase])
    assert lote.shape == fase.shape and lote.dtype == np.float32
    np.testing.assert_allclose(lote, referencia, atol=1e-4)


def test_sanitizar_fase_lote_sem_metade_igual_a_versao_por_quadro():
    fase = _fase_sintetica((3, 114, 10), seed=1)
    np.testing.assert_allclose(
        sanitizar_fase_lote(fase, por_metade=False),
        sanitizar_fase(fase, por_metade=False),
        atol=1e-4,
    )


def test_sanitizar_fase_lote_remove_rampa_pura():
    k = np.arange(114, dtype=np.float64)[:, None] * np.ones((1, 10))
    fase = np.angle(np.exp(1j * (0.2 * k + 0.5))).astype(np.float32)[None]
    assert np.abs(sanitizar_fase_lote(fase)).max() < 1e-4


# ==========================================================================
# Protocolos
# ==========================================================================

def test_listar_sequencias_cobre_tudo():
    todas = listar_sequencias()
    assert len(todas) == 1080 and len(set(todas)) == 1080
    assert len(listar_sequencias(PROTOCOLO_1)) == 14 * 40
    with pytest.raises(ValueError):
        listar_sequencias(["A99"])


def test_s2_usa_exatamente_os_sujeitos_oficiais():
    treino, teste = dividir_s2()
    assert not set(treino) & set(teste)
    assert {s for s, _ in teste} == set(SUJEITOS_TESTE_S2)
    assert not {s for s, _ in treino} & set(SUJEITOS_TESTE_S2)
    assert len(treino) + len(teste) == 1080
    assert len(teste) == 8 * 27


def test_s1_disjunto_estratificado_e_deterministico():
    treino, teste = dividir_s1_agrupado(PROTOCOLO_3, seed=0)
    assert not set(treino) & set(teste)
    assert len(treino) + len(teste) == 1080
    assert len(teste) == 270
    por_acao = {a: sum(1 for _, b in teste if b == a) for a in PROTOCOLO_3}
    assert set(por_acao.values()) == {10}

    assert dividir_s1_agrupado(PROTOCOLO_3, seed=0) == (treino, teste)
    assert dividir_s1_agrupado(PROTOCOLO_3, seed=1)[1] != teste


def test_s1_sorteio_por_acao_independe_do_conjunto():
    """A mesma seed sorteia os mesmos sujeitos para A02 em P1 e em P3."""
    _, teste_p3 = dividir_s1_agrupado(PROTOCOLO_3, seed=7)
    _, teste_p1 = dividir_s1_agrupado(PROTOCOLO_1, seed=7)
    assert {s for s, a in teste_p3 if a == "A02"} == {s for s, a in teste_p1 if a == "A02"}


def test_s1_proporcao_invalida():
    for ruim in (0.0, 1.0, -0.1, 0.001):
        with pytest.raises(ValueError):
            dividir_s1_agrupado(PROTOCOLO_3, proporcao_teste=ruim)


# ==========================================================================
# Janelas e atributos
# ==========================================================================

def test_janelas_padrao_sao_9_sem_sobreposicao():
    janelas = janelas_de_sequencia(297, 30)
    assert len(janelas) == 9
    assert janelas[0] == (0, 30) and janelas[-1] == (240, 270)
    assert janelas_de_sequencia(297, 297) == [(0, 297)]
    assert len(janelas_de_sequencia(297, 30, 15)) == 18


@pytest.mark.parametrize("ruim", [0, -1, 298])
def test_janela_invalida_cita_a_faixa(ruim):
    with pytest.raises(ValueError, match="1 a 297"):
        janelas_de_sequencia(297, ruim)


def test_extrair_atributos_valores_e_ordem():
    gerador = np.random.default_rng(0)
    amp = gerador.normal(20, 2, size=(30, 3, 114, 10)).astype(np.float32)
    fase = gerador.normal(0, 1, size=(30, 3, 114, 10)).astype(np.float32)
    x = extrair_atributos(amp, fase)
    assert x.shape == (N_ATRIBUTOS,) == (1368,)

    # Antena 2, subportadora 5 -> posição 2*114 + 5 em cada bloco de 342.
    p = 2 * 114 + 5
    traco_a = amp[:, 2, 5, :].astype(np.float64)
    traco_f = fase[:, 2, 5, :].astype(np.float64)
    assert x[p] == pytest.approx(traco_a.mean(), rel=1e-5)
    assert x[342 + p] == pytest.approx(traco_a.std(), rel=1e-5)
    assert x[684 + p] == pytest.approx(traco_f.mean(), abs=1e-5)
    assert x[1026 + p] == pytest.approx(traco_f.std(), rel=1e-5)

    lote = extrair_atributos(np.stack([amp, amp + 1]), np.stack([fase, fase]))
    assert lote.shape == (2, 1368)
    np.testing.assert_allclose(lote[1, :342], x[:342] + 1, rtol=1e-5)
    np.testing.assert_allclose(lote[1, 342:], x[342:], rtol=1e-4, atol=1e-5)


def test_montar_conjunto_em_cache_sintetico():
    sequencias = [("S01", "A01"), ("S02", "A02")]
    amp = np.zeros((2, 297, 3, 114, 10), dtype=np.float32)
    amp[1] = 5.0
    fase = np.zeros_like(amp)
    linhas = {("S01", "A01"): 0, ("S02", "A02"): 1}
    X, y, grupos = montar_conjunto(amp, fase, linhas, sequencias[::-1],
                                   ("A01", "A02"), janela=30)
    assert X.shape == (18, 1368)
    assert list(y) == [1] * 9 + [0] * 9
    assert list(grupos) == [0] * 9 + [1] * 9
    assert np.all(X[:9, :342] == 5.0) and np.all(X[9:, :342] == 0.0)


# ==========================================================================
# Métricas
# ==========================================================================

def test_agregar_por_sequencia_media_das_probabilidades():
    # Sequência 0: duas janelas erradas com pouca confiança, uma certa com muita.
    proba = np.array([[0.4, 0.6], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 0, 0, 1])
    grupos = np.array([0, 0, 0, 1])
    y_seq, pred_seq, ids = agregar_por_sequencia(proba, y, grupos)
    assert list(ids) == [0, 1]
    assert list(y_seq) == [0, 1]
    assert list(pred_seq) == [0, 1]  # média 0.567 vs 0.433 para a classe 0


def test_agregar_rejeita_rotulos_misturados():
    with pytest.raises(ValueError):
        agregar_por_sequencia(np.eye(2), np.array([0, 1]), np.array([0, 0]))


def test_ic_bootstrap_contem_a_media_e_e_deterministico():
    acertos = np.array([1] * 70 + [0] * 30)
    baixo, alto = ic_bootstrap(acertos, seed=0)
    assert baixo < 0.7 < alto
    assert 0.55 < baixo and alto < 0.85
    assert ic_bootstrap(acertos, seed=0) == (baixo, alto)
    assert ic_bootstrap(np.ones(10)) == (1.0, 1.0)


def test_ic_bootstrap_por_grupo_e_mais_largo():
    """Janelas da mesma sequência acertam ou erram juntas: reamostrar janelas
    soltas subestimaria a incerteza."""
    acertos_seq = np.array([1] * 14 + [0] * 6)
    acertos = np.repeat(acertos_seq, 9)
    grupos = np.repeat(np.arange(20), 9)
    b_grupo, a_grupo = ic_bootstrap(acertos, grupos, seed=0)
    b_solto, a_solto = ic_bootstrap(acertos, None, seed=0)
    assert (a_grupo - b_grupo) > 2 * (a_solto - b_solto)


def test_avaliar_estrutura_e_valores():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y = np.array([0, 0, 1, 1])
    grupos = np.array([0, 0, 1, 1])
    r = avaliar(proba, y, grupos, n_classes=2)
    assert r["por_janela"]["acuracia"] == 0.75
    assert r["por_sequencia"]["acuracia"] == 1.0  # média [0.45, 0.55] -> classe 1
    assert r["acaso"] == 0.5
    assert r["por_janela"]["confusao"] == [[2, 0], [1, 1]]
    assert r["por_sequencia"]["confusao"] == [[1, 0], [0, 1]]


def test_avaliar_ic_por_janela_reamostra_sequencias_inteiras():
    """Se avaliar passasse grupos=None ao bootstrap por janela, o IC encolheria."""
    acertos_seq = np.array([1] * 14 + [0] * 6)
    y = np.repeat(np.arange(20) % 2, 9)
    grupos = np.repeat(np.arange(20), 9)
    acertos = np.repeat(acertos_seq, 9)
    pred = np.where(acertos == 1, y, 1 - y)
    proba = np.eye(2)[pred] * 0.8 + 0.1
    r = avaliar(proba, y, grupos, n_classes=2, seed=3)

    ic = r["por_janela"]["ic95"]
    assert ic == ic_bootstrap(acertos, grupos, seed=3)
    b_solto, a_solto = ic_bootstrap(acertos, None, seed=3)
    assert (ic[1] - ic[0]) > 2 * (a_solto - b_solto)

    y_seq = np.arange(20) % 2
    pred_seq = np.where(acertos_seq == 1, y_seq, 1 - y_seq)
    esperada = np.zeros((2, 2), dtype=int)
    np.add.at(esperada, (y_seq, pred_seq), 1)
    assert r["por_sequencia"]["confusao"] == esperada.tolist()


# ==========================================================================
# Cache (sem dataset: só a lógica do marcador)
# ==========================================================================

def test_cache_sem_marcador_e_incompleto(tmp_path):
    for nome in ("amp.npy", "fase.npy", "indice.json"):
        (tmp_path / nome).write_text("x")
    assert not cache_csi.cache_completo(tmp_path)
    with pytest.raises(FileNotFoundError):
        cache_csi.abrir_cache(tmp_path)


def test_cache_com_dataset_ausente_cita_a_variavel(tmp_path):
    with pytest.raises(FileNotFoundError, match="MMFI_DATASET"):
        cache_csi.construir_cache(tmp_path / "cache", raiz=tmp_path / "nada")


def _cache_sintetico(tmp_path, monkeypatch):
    """Cache de 3 sequências sintéticas, sem tocar no disco do dataset."""
    poucas = [("S01", "A01"), ("S02", "A05"), ("S07", "A27")]
    dados = {}
    for i, seq in enumerate(poucas):
        gerador = np.random.default_rng(i)
        amp_seq = gerador.normal(20, 3, size=(297, 3, 114, 10)).astype(np.float32)
        dados[seq] = (amp_seq, _fase_sintetica((297, 3, 114, 10), seed=i))
    monkeypatch.setattr(cache_csi, "listar_sequencias", lambda: list(poucas))
    monkeypatch.setattr(cache_csi, "formato_disponivel", lambda s, a, r=None: "pacote")
    monkeypatch.setattr(cache_csi, "carregar_csi_sequencia",
                        lambda s, a, r=None: dados[(s, a)])
    return poucas, dados


def test_cache_sintetico_ida_e_volta(tmp_path, monkeypatch):
    poucas, dados = _cache_sintetico(tmp_path, monkeypatch)
    destino = cache_csi.construir_cache(tmp_path / "cache", raiz=tmp_path)
    amp, fase, linhas = cache_csi.abrir_cache(destino)
    assert amp.shape == (3, 297, 3, 114, 10)
    assert set(linhas) == set(poucas)
    for seq, (amp_ref, fase_ref) in dados.items():
        np.testing.assert_array_equal(amp[linhas[seq]], amp_ref)
        np.testing.assert_array_equal(fase[linhas[seq]], sanitizar_fase_lote(fase_ref))


@pytest.mark.parametrize("campo, valor", [
    ("versao", cache_csi.VERSAO_CACHE + 1),
    ("sequencias", [["S01", "A01"]]),
])
def test_cache_com_indice_desatualizado_e_refeito(tmp_path, monkeypatch, capsys,
                                                 campo, valor):
    _cache_sintetico(tmp_path, monkeypatch)
    destino = cache_csi.construir_cache(tmp_path / "cache", raiz=tmp_path)
    arquivo = destino / "indice.json"
    indice = json.loads(arquivo.read_text(encoding="utf-8"))
    indice[campo] = valor
    arquivo.write_text(json.dumps(indice), encoding="utf-8")
    capsys.readouterr()

    cache_csi.construir_cache(destino, raiz=tmp_path)
    assert "desatualizado" in capsys.readouterr().out
    refeito = json.loads(arquivo.read_text(encoding="utf-8"))
    assert refeito["versao"] == cache_csi.VERSAO_CACHE
    assert len(refeito["sequencias"]) == 3


def test_cache_valido_ignora_raiz_diferente(tmp_path, monkeypatch, capsys):
    """Mover a pasta do dataset não pode forçar reconstrução."""
    _cache_sintetico(tmp_path, monkeypatch)
    destino = cache_csi.construir_cache(tmp_path / "cache", raiz=tmp_path)
    marca = (destino / "amp.npy").stat().st_mtime_ns
    capsys.readouterr()
    cache_csi.construir_cache(destino, raiz=tmp_path / "outro_lugar")
    assert (destino / "amp.npy").stat().st_mtime_ns == marca
    assert "AVISO" not in capsys.readouterr().out


# ==========================================================================
# Cache com dataset — um cache reduzido de 2 sequências
# ==========================================================================

@pytest.fixture
def cache_pequeno(tmp_path, monkeypatch):
    poucas = [("S01", "A01"), ("S10", "A03")]
    monkeypatch.setattr(cache_csi, "listar_sequencias", lambda: poucas)
    destino = cache_csi.construir_cache(tmp_path / "cache")
    return destino, poucas


@precisa_dataset
def test_cache_bate_com_carregar_csi_sequencia(cache_pequeno):
    destino, poucas = cache_pequeno
    amp, fase, linhas = cache_csi.abrir_cache(destino)
    assert isinstance(amp, np.memmap)
    assert amp.shape == (2, 297, 3, 114, 10) and amp.dtype == np.float32
    for seq in poucas:
        amp_ref, fase_ref = carregar_csi_sequencia(*seq)
        np.testing.assert_array_equal(amp[linhas[seq]], amp_ref)
        # Quadro avulso conferido contra a sanitização original, por quadro.
        np.testing.assert_allclose(fase[linhas[seq]][100], sanitizar_fase(fase_ref[100]),
                                   atol=1e-4)
    assert np.isfinite(amp).all()


@precisa_dataset
def test_cache_completo_e_reaproveitado_e_incompleto_e_refeito(cache_pequeno, capsys):
    destino, poucas = cache_pequeno
    marca = (destino / "amp.npy").stat().st_mtime_ns
    cache_csi.construir_cache(destino)
    assert (destino / "amp.npy").stat().st_mtime_ns == marca  # não releu

    (destino / cache_csi.MARCADOR).unlink()
    cache_csi.construir_cache(destino)
    assert "incompleto" in capsys.readouterr().out
    assert cache_csi.cache_completo(destino)
    indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
    assert [tuple(s) for s in indice["sequencias"]] == poucas
