"""Testes unitários para a arquitetura da Rede Neural Multi-Head de CSI."""

import numpy as np
import pytest
import torch

from src.rede_neural import CSISpatialSpectralTemporalNet, calcular_mpjpe


def test_instanciacao_da_rede_padrao():
    """Garante que a rede pode ser instanciada com os parâmetros padrão do MM-Fi."""
    modelo = CSISpatialSpectralTemporalNet(in_channels=3, num_classes=27, num_joints=17)
    assert modelo.in_channels == 3
    assert modelo.num_classes == 27
    assert modelo.num_joints == 17


def test_forward_pass_com_amplitude_pura():
    """Testa o passo forward com lote de amplitude (3 canais: 3 antenas)."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 114, 10)
    modelo = CSISpatialSpectralTemporalNet(in_channels=3, num_classes=27, num_joints=17)
    modelo.eval()

    with torch.no_grad():
        saida = modelo(x)

    assert "logits" in saida
    assert "pose_3d" in saida
    assert "embedding" in saida

    assert saida["logits"].shape == (batch_size, 27)
    assert saida["pose_3d"].shape == (batch_size, 17, 3)
    assert saida["embedding"].shape == (batch_size, modelo.latent_dim)


def test_forward_pass_com_amplitude_e_fase():
    """Testa o passo forward com 6 canais (3 amplitude + 3 fase sanitizada)."""
    batch_size = 2
    x = torch.randn(batch_size, 6, 114, 10)
    modelo = CSISpatialSpectralTemporalNet(in_channels=6, num_classes=27, num_joints=17)
    modelo.eval()

    with torch.no_grad():
        saida = modelo(x)

    assert saida["logits"].shape == (batch_size, 27)
    assert saida["pose_3d"].shape == (batch_size, 17, 3)


def test_retropropagacao_e_gradientes():
    """Verifica se o gradiente flui perfeitamente para ambas as cabeças de perda."""
    modelo = CSISpatialSpectralTemporalNet(in_channels=3, num_classes=27, num_joints=17)
    modelo.train()

    x = torch.randn(2, 3, 114, 10, requires_grad=True)
    rotulos = torch.tensor([0, 15], dtype=torch.long)
    poses_reais = torch.randn(2, 17, 3)

    saida = modelo(x)
    loss_cls = torch.nn.functional.cross_entropy(saida["logits"], rotulos)
    loss_pose = torch.nn.functional.mse_loss(saida["pose_3d"], poses_reais)
    loss_total = loss_cls + 10.0 * loss_pose

    loss_total.backward()

    # Verifica se os gradientes foram calculados nos pesos do backbone e cabeças
    for nome, param in modelo.named_parameters():
        assert param.grad is not None, f"Gradiente ausente no parâmetro {nome}"
        assert not torch.isnan(param.grad).any(), f"NaN detectado no gradiente de {nome}"


def test_ablacao_de_subportadoras_modifica_saida():
    """Tarefa (e): mascara de subportadoras deve alterar o resultado da predição."""
    torch.manual_seed(42)
    modelo = CSISpatialSpectralTemporalNet(in_channels=3, num_classes=27, num_joints=17)
    modelo.eval()

    x = torch.randn(2, 3, 114, 10)

    with torch.no_grad():
        saida_completa = modelo(x)

        # Máscara zerando metade das subportadoras (114 subportadoras)
        mascara = torch.ones(114)
        mascara[50:] = 0.0
        saida_ablada = modelo(x, mascara_subportadoras=mascara)

    diferenca = torch.norm(saida_completa["logits"] - saida_ablada["logits"])
    assert diferenca.item() > 1e-4, "Ablação de subportadoras não produziu efeito perceptível."


def test_calculo_mpjpe_correto():
    """Testa se o cálculo do MPJPE em milímetros funciona conforme a literatura."""
    # Duas poses idênticas devem ter erro 0.0 mm
    p1 = np.ones((1, 17, 3), dtype=np.float32)
    assert calcular_mpjpe(p1, p1) == 0.0

    # Deslocamento de 0.1 metro (10 cm = 100 mm) em todas as juntas
    p2 = p1 + 0.1
    # Distância Euclidiana por junta = sqrt(0.1^2 + 0.1^2 + 0.1^2) = 0.1 * sqrt(3) m = 173.205 mm
    esperado = 1000.0 * 0.1 * np.sqrt(3)  # 173.205 mm
    obtido = calcular_mpjpe(p1, p2)
    assert np.isclose(obtido, esperado, atol=1e-2)
