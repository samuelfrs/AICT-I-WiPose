"""
Testes do carregamento do MM-Fi (src/mmfi.py).

Divididos em dois grupos:

* Os que **não precisam do dataset** — validação de argumentos, constantes do
  esqueleto e o comportamento de ``sanitizar_fase`` sobre sinal sintético. Rodam
  em qualquer máquina logo após clonar o repositório.
* Os marcados com ``@pytest.mark.dataset`` — precisam do MM-Fi em disco e são
  pulados automaticamente se ele não estiver lá.

A ideia é testar **valores**, não formas: conferir ``csi.shape == (3,114,10)``
usando as mesmas constantes do módulo não detecta eixo trocado. Onde importa, os
testes comparam contra o arquivo lido diretamente com ``scipy.io.loadmat``.

Uso:
    .venv/Scripts/python.exe -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mmfi import (  # noqa: E402
    ACOES,
    CONEXOES_ESQUELETO,
    FRONTEIRA_20MHZ,
    N_JUNTAS,
    N_QUADROS,
    N_SUBPORTADORAS,
    PROTOCOLO_1,
    PROTOCOLO_2,
    ambiente_do_sujeito,
    caminho_dataset,
    caminho_sequencia,
    carregar_pose_sequencia,
    carregar_quadro,
    sanitizar_fase,
)

TEM_DATASET = (caminho_dataset() / "E01" / "S01" / "A01").is_dir()
precisa_dataset = pytest.mark.skipif(
    not TEM_DATASET,
    reason="MM-Fi não encontrado. Aponte a variável de ambiente MMFI_DATASET.",
)


# ==========================================================================
# Sem dataset — constantes e validação
# ==========================================================================

def test_as_27_acoes_estao_declaradas():
    assert len(ACOES) == 27
    assert sorted(ACOES) == [f"A{i:02d}" for i in range(1, 28)]


def test_protocolos_particionam_as_acoes():
    # P1 (diárias) + P2 (reabilitação) = as 27, sem sobreposição.
    assert len(PROTOCOLO_1) == 14
    assert len(PROTOCOLO_2) == 13
    assert not set(PROTOCOLO_1) & set(PROTOCOLO_2)
    assert set(PROTOCOLO_1) | set(PROTOCOLO_2) == set(ACOES)


def test_sujeitos_mapeiam_para_o_ambiente_certo():
    # 10 sujeitos por ambiente, em blocos contíguos.
    assert ambiente_do_sujeito("S01") == "E01"
    assert ambiente_do_sujeito("S10") == "E01"
    assert ambiente_do_sujeito("S11") == "E02"
    assert ambiente_do_sujeito("S30") == "E03"
    assert ambiente_do_sujeito("S31") == "E04"
    assert ambiente_do_sujeito("S40") == "E04"


def test_sujeito_malformado_falha_com_mensagem_clara():
    # Sem a validação, "S1" viraria um caminho inexistente e o erro culparia o
    # caminho do dataset — o lugar errado para procurar o problema.
    for ruim in ("S1", "s10", "S 5", "SXX", "S100", ""):
        with pytest.raises(ValueError, match="inválido|intervalo"):
            ambiente_do_sujeito(ruim)


def test_sujeito_fora_do_intervalo_falha():
    with pytest.raises(ValueError, match="intervalo"):
        ambiente_do_sujeito("S41")
    with pytest.raises(ValueError, match="intervalo"):
        ambiente_do_sujeito("S00")


def test_acao_desconhecida_falha():
    with pytest.raises(ValueError, match="Ação desconhecida"):
        caminho_sequencia("S01", "A99")


def test_esqueleto_tem_16_ossos_ligando_as_17_juntas():
    assert len(CONEXOES_ESQUELETO) == N_JUNTAS - 1  # árvore, sem ciclos
    indices = {i for osso in CONEXOES_ESQUELETO for i in osso}
    assert indices == set(range(N_JUNTAS)), "há junta solta no esqueleto"


def test_esqueleto_e_uma_arvore_enraizada_no_quadril():
    # Toda junta tem que ser alcançável a partir da raiz (junta 0).
    vizinhos = {i: set() for i in range(N_JUNTAS)}
    for a, b in CONEXOES_ESQUELETO:
        vizinhos[a].add(b)
        vizinhos[b].add(a)

    visitados, fila = {0}, [0]
    while fila:
        atual = fila.pop()
        for v in vizinhos[atual] - visitados:
            visitados.add(v)
            fila.append(v)
    assert visitados == set(range(N_JUNTAS))


# ==========================================================================
# Sem dataset — sanitização de fase sobre sinal sintético
# ==========================================================================

def _fase_sintetica(inclinacao_a: float, inclinacao_b: float) -> np.ndarray:
    """Fase com uma rampa diferente em cada bloco de 20 MHz, envolvida em ±π."""
    k = np.arange(N_SUBPORTADORAS, dtype=np.float64)
    limpa = np.where(k < FRONTEIRA_20MHZ,
                     inclinacao_a * k,
                     inclinacao_b * (k - FRONTEIRA_20MHZ) + 2.0)
    envolvida = np.angle(np.exp(1j * limpa))  # simula o wrap em ±π
    return np.tile(envolvida[None, :, None], (3, 1, 10)).astype(np.float32)


def test_sanitizar_remove_rampas_diferentes_em_cada_bloco():
    fase = _fase_sintetica(0.05, -0.08)
    saneada = sanitizar_fase(fase, por_metade=True)
    # Removida a rampa de cada bloco, sobra essencialmente zero.
    assert np.abs(saneada).max() < 1e-3


def test_ajuste_por_metade_bate_o_ajuste_unico_quando_ha_descontinuidade():
    # É esta a justificativa do FRONTEIRA_20MHZ: com rampas diferentes nos dois
    # blocos, uma reta só sobre as 114 deixa resíduo grande.
    fase = _fase_sintetica(0.05, -0.08)
    residuo_metade = sanitizar_fase(fase, por_metade=True).std()
    residuo_unico = sanitizar_fase(fase, por_metade=False).std()
    assert residuo_metade < residuo_unico / 10


def test_sanitizar_avisa_quando_nao_consegue_dividir():
    fase = np.zeros((3, 64, 10), dtype=np.float32)  # não são 114 subportadoras
    with pytest.warns(UserWarning, match="por_metade"):
        sanitizar_fase(fase, por_metade=True)


def test_sanitizar_rejeita_tensor_com_dimensao_errada():
    with pytest.raises(ValueError, match="ndim"):
        sanitizar_fase(np.zeros((114, 10), dtype=np.float32))


# ==========================================================================
# Com dataset
# ==========================================================================

@precisa_dataset
def test_carregar_quadro_devolve_as_formas_do_contrato():
    amplitude, fase, juntas, meta = carregar_quadro("S01", "A01", 1)
    assert amplitude.shape == (3, 114, 10)  # literais de propósito
    assert fase.shape == (3, 114, 10)
    assert juntas.shape == (17, 3)
    assert meta["sujeito"] == "S01" and meta["ambiente"] == "E01"


@precisa_dataset
def test_quadro_e_1_based_e_casa_com_a_sequencia():
    """O erro clássico aqui é off-by-one: os arquivos são frame001..frame297,
    mas o ground_truth é indexado de 0. Este teste fixa a correspondência."""
    sequencia = carregar_pose_sequencia("S01", "A01")
    for quadro in (1, 42, N_QUADROS):
        _, _, juntas, _ = carregar_quadro("S01", "A01", quadro)
        assert np.array_equal(juntas, sequencia[quadro - 1].astype(np.float32))
        if quadro < N_QUADROS:
            assert not np.array_equal(juntas, sequencia[quadro])


@precisa_dataset
def test_quadro_fora_do_intervalo_falha():
    for ruim in (0, -1, N_QUADROS + 1):
        with pytest.raises(ValueError, match="intervalo"):
            carregar_quadro("S01", "A01", ruim)


@precisa_dataset
def test_amplitude_bate_com_a_leitura_direta_do_arquivo():
    """Compara contra scipy.io.loadmat cru, para pegar transposição ou reshape
    que embaralhe os eixos sem levantar exceção."""
    import scipy.io as sio

    amplitude, fase, _, _ = carregar_quadro("S01", "A01", 1)
    bruto = sio.loadmat(
        caminho_sequencia("S01", "A01") / "wifi-csi" / "frame001.mat"
    )
    assert np.allclose(amplitude, bruto["CSIamp"].astype(np.float32))
    assert np.allclose(fase, bruto["CSIphase"].astype(np.float32))


@precisa_dataset
def test_nao_finitos_sao_corrigidos_sem_tocar_nos_valores_validos():
    """E01/S10/A03/frame042 tem um -inf conhecido em CSIamp."""
    import scipy.io as sio

    amplitude, _, _, meta = carregar_quadro("S10", "A03", 42)
    bruto = sio.loadmat(
        caminho_sequencia("S10", "A03") / "wifi-csi" / "frame042.mat"
    )["CSIamp"]

    assert meta["valores_corrigidos"] > 0, "esperado ao menos um -inf aqui"
    assert np.isfinite(amplitude).all(), "sobrou valor não-finito"
    validos = np.isfinite(bruto)
    assert np.allclose(amplitude[validos], bruto[validos].astype(np.float32))


@precisa_dataset
def test_fase_e_crua_e_amplitude_e_dB():
    _, fase, _, _ = carregar_quadro("S01", "A01", 1)
    # Fase crua encosta em ±π; se um dia vier saneada, este teste avisa.
    assert fase.min() < -3.0 and fase.max() > 3.0

    amplitude, _, _, _ = carregar_quadro("S01", "A01", 1)
    # Em dB os valores ficam na casa das dezenas; em escala linear seriam ~0-1.
    assert 1.0 < amplitude.mean() < 100.0


@precisa_dataset
def test_o_eixo_y_cresce_para_baixo_e_a_unidade_e_metro():
    """Verificação anatômica: com o esqueleto Human3.6M, cabeça (10) fica ACIMA
    dos pés (3, 6). Como y cresce para baixo, a cabeça tem y MENOR."""
    _, _, juntas, _ = carregar_quadro("S01", "A01", 1)
    cabeca, pes = juntas[10], juntas[[3, 6]].mean(axis=0)

    assert cabeca[1] < pes[1], "y deveria crescer para baixo"
    altura = float(np.linalg.norm(cabeca - pes))
    assert 1.0 < altura < 2.0, f"vão cabeça→pés de {altura:.2f} não parece metro"


@precisa_dataset
def test_juntas_esquerda_e_direita_nao_estao_trocadas():
    """Em A13 ('levantar mão esquerda') quem sobe é o punho esquerdo (13)."""
    poses = carregar_pose_sequencia("S01", "A13")
    # Menor y = mais alto. Comparamos o ponto mais alto que cada punho atinge.
    alcance_esq = poses[:, 13, 1].min()
    alcance_dir = poses[:, 16, 1].min()
    assert alcance_esq < alcance_dir, "punho esquerdo deveria subir mais em A13"
