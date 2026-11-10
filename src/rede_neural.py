"""Rede Neural Residual Espectro-Temporal para Classificação de Ações via WiFi CSI.

Arquitetura temporal otimizada para o dataset MM-Fi:
- Entrada: Vetor espectro-temporal de 1.368 descritores por janela de 30 quadros (3 segundos):
    - [amp_media (342) | amp_desvio (342) | fase_media (342) | fase_desvio (342)]
    - Onde 342 = 3 antenas receptoras × 114 subportadoras OFDM.
- Backbone: Projeção linear com blocos residuais profundos, Batch Normalization, GELU e Dropout.
- Tarefa (c): Classificação discriminativa de ações físicas (P1: 14 ações, P2: 13 ações, P3: 27 ações).
- Tarefa (e): Suporte a ablação de subportadoras OFDM ruidosas via máscara espectral.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object
    F = None


def aplicar_mascara_subportadoras(
    x: "torch.Tensor", mascara: "torch.Tensor"
) -> "torch.Tensor":
    """Aplica máscara de ablação OFDM (114,) sobre o vetor de características (B, 1368).

    O vetor x possui 1.368 dimensões organizadas em 4 blocos de (3 antenas × 114 subportadoras):
    - [amp_media (342), amp_desvio (342), fase_media (342), fase_desvio (342)].
    A máscara (114,) modula cada bloco de 342 = 3 × 114, zerando subportadoras abladas.
    """
    if mascara is None:
        return x
    if mascara.dim() != 1 or mascara.shape[0] != 114:
        raise ValueError(
            f"Máscara de subportadoras deve ter formato (114,), recebido {mascara.shape}."
        )
    # Expande máscara de 114 para 3 antenas -> shape (342,)
    mascara_bloco = mascara.view(1, 114).repeat(3, 1).reshape(-1)
    # Repete para os 4 blocos estatísticos -> shape (1, 1368)
    mascara_total = mascara_bloco.repeat(4).view(1, -1).to(x.device)
    return x * mascara_total


class ResidualTemporalBlock(nn.Module if torch else object):
    """Bloco residual profundo com Batch Normalization, GELU e Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.3):
        if torch is None:
            raise RuntimeError("PyTorch não está instalado no ambiente.")
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class CSITemporalActionNet(nn.Module if torch else object):
    """Rede Neural Residual Espectro-Temporal para Reconhecimento de Ações via WiFi CSI.

    Parâmetros:
    -----------
    in_features : int
        Número de características espectro-temporais (padrão: 1368 = 4 blocos × 3 antenas × 114 subportadoras).
    num_classes : int
        Número de classes de ações (14 para P1, 13 para P2, 27 para P3).
    hidden_dim : int
        Dimensão do espaço latente intermediário (padrão: 512).
    num_blocks : int
        Número de blocos residuais profundos no backbone (padrão: 3).
    dropout_rate : float
        Taxa de dropout para regularização e prevenção de overfitting (padrão: 0.3).
    """

    def __init__(
        self,
        in_features: int = 1368,
        num_classes: int = 27,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        dropout_rate: float = 0.3,
    ):
        if torch is None:
            raise RuntimeError("PyTorch não está instalado no ambiente.")
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. Projeção Linear de Entrada com Normalização
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Backbone Residual Profundo
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )

        # 3. Cabeça de Classificação de Ações
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: "torch.Tensor",
        mascara_subportadoras: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Passo forward da rede.

        Parâmetros:
        -----------
        x : torch.Tensor
            Tensor com dimensões (Batch, in_features).
        mascara_subportadoras : torch.Tensor, opcional
            Máscara de dimensão (114,) para simular ablação de subportadoras OFDM.
        """
        if mascara_subportadoras is not None:
            x = aplicar_mascara_subportadoras(x, mascara_subportadoras)

        embedding = self.in_proj(x)
        for block in self.blocks:
            embedding = block(embedding)

        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "embedding": embedding,
        }


# Mantém compatibilidade com funções utilitárias prévias
def calcular_mpjpe(
    pose_pred: Union["torch.Tensor", np.ndarray],
    pose_real: Union["torch.Tensor", np.ndarray],
) -> float:
    """Calcula o Mean Per Joint Position Error (MPJPE) em milímetros (mm)."""
    if hasattr(pose_pred, "detach"):
        pose_pred = pose_pred.detach().cpu().numpy()
    if hasattr(pose_real, "detach"):
        pose_real = pose_real.detach().cpu().numpy()

    dists = np.linalg.norm(pose_pred - pose_real, axis=-1)
    return float(np.mean(dists) * 1000.0)
