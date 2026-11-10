"""Testes unitários para a arquitetura da Rede Neural Residual Temporal de CSI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset_rede import MMFiTemporalDataset
from src.rede_neural import (
    CSITemporalActionNet,
    aplicar_mascara_subportadoras,
    calcular_mpjpe,
)
from scripts.treinar_rede import avaliar_rede


def test_instanciacao_da_rede_padrao():
    """Garante que a rede temporal pode ser instanciada com os parâmetros padrão do MM-Fi."""
    modelo = CSITemporalActionNet(in_features=1368, num_classes=27, hidden_dim=512, num_blocks=3)
    assert modelo.in_features == 1368
    assert modelo.num_classes == 27
    assert modelo.hidden_dim == 512


def test_forward_pass_temporal():
    """Testa o passo forward com lote de descritores espectro-temporais (B, 1368)."""
    batch_size = 4
    x = torch.randn(batch_size, 1368)
    modelo = CSITemporalActionNet(in_features=1368, num_classes=27, hidden_dim=256, num_blocks=2)
    modelo.eval()

    with torch.no_grad():
        saida = modelo(x)

    assert "logits" in saida
    assert "embedding" in saida
    assert saida["logits"].shape == (batch_size, 27)
    assert saida["embedding"].shape == (batch_size, 256)


def test_retropropagacao_e_gradientes():
    """Verifica se o gradiente flui perfeitamente através de todos os blocos residuais."""
    modelo = CSITemporalActionNet(in_features=1368, num_classes=27, hidden_dim=256, num_blocks=2)
    modelo.train()

    x = torch.randn(8, 1368, requires_grad=True)
    rotulos = torch.tensor([0, 5, 10, 15, 20, 25, 1, 2], dtype=torch.long)

    saida = modelo(x)
    loss = torch.nn.functional.cross_entropy(saida["logits"], rotulos)
    loss.backward()

    for nome, param in modelo.named_parameters():
        assert param.grad is not None, f"Gradiente ausente no parâmetro {nome}"
        assert not torch.isnan(param.grad).any(), f"NaN detectado no gradiente de {nome}"


def test_mascara_subportadoras_fisica():
    """Verifica se a máscara de 114 subportadoras modula perfeitamente os 4 blocos de 342."""
    x = torch.ones(2, 1368)
    mascara = torch.ones(114)
    mascara[10:] = 0.0  # Mantém apenas as 10 primeiras subportadoras

    x_ablado = aplicar_mascara_subportadoras(x, mascara)
    assert x_ablado.shape == (2, 1368)

    # Cada bloco de 342 = 3 antenas * 114 subportadoras
    # As 10 primeiras subportadoras das 3 antenas = 30 valores ativos por bloco
    # 4 blocos * 30 = 120 valores ativos no total
    ativos = (x_ablado[0] > 0).sum().item()
    assert ativos == 120


def test_ablacao_de_subportadoras_modifica_saida():
    """Tarefa (e): máscara de subportadoras deve alterar o resultado da predição."""
    torch.manual_seed(42)
    modelo = CSITemporalActionNet(in_features=1368, num_classes=27, hidden_dim=256, num_blocks=2)
    modelo.eval()

    x = torch.randn(2, 1368)

    with torch.no_grad():
        saida_completa = modelo(x)

        mascara = torch.ones(114)
        mascara[50:] = 0.0
        saida_ablada = modelo(x, mascara_subportadoras=mascara)

    diferenca = torch.norm(saida_completa["logits"] - saida_ablada["logits"])
    assert diferenca.item() > 1e-4, "Ablação de subportadoras não produziu efeito perceptível."


def test_dataset_e_soft_voting():
    """Verifica o carregamento do dataset e o cálculo correto do Soft Voting por sequência."""
    # Instancia pequeno dataset para teste
    ds_tr = MMFiTemporalDataset(split="S2", fase="treino", protocolo="P1")
    ds_te = MMFiTemporalDataset(
        split="S2", fase="teste", protocolo="P1", media_scaler=ds_tr.media, desvio_scaler=ds_tr.desvio
    )

    assert ds_tr.num_classes == 14
    assert ds_te.num_classes == 14
    assert len(ds_tr) > 0
    assert len(ds_te) > 0

    item = ds_te[0]
    assert item["csi"].shape == (1368,)
    assert 0 <= item["rotulo_acao"].item() < 14

    dl = DataLoader(ds_te, batch_size=32, shuffle=False)
    modelo = CSITemporalActionNet(in_features=1368, num_classes=14, hidden_dim=128, num_blocks=1)

    device = torch.device("cpu")
    res = avaliar_rede(modelo, dl, device)

    assert "acc_janela" in res
    assert "acc_seq" in res
    assert 0.0 <= res["acc_janela"] <= 1.0
    assert 0.0 <= res["acc_seq"] <= 1.0


def test_calculo_mpjpe_correto():
    """Testa se o cálculo do MPJPE em milímetros se mantém funcional."""
    p1 = np.ones((1, 17, 3), dtype=np.float32)
    assert calcular_mpjpe(p1, p1) == 0.0

    p2 = p1 + 0.1
    # Distância Euclidiana em cada junta = sqrt(0.1^2 + 0.1^2 + 0.1^2) = 0.1 * sqrt(3) ~ 0.1732 m = 173.2 mm
    mpjpe = calcular_mpjpe(p1, p2)
    assert np.isclose(mpjpe, 173.205, atol=0.1)
