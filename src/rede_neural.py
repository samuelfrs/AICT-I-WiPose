"""Rede Neural Multi-Head para Classificação de Ações e Estimação de Pose 3D via WiFi CSI.

Arquitetura convolucional espaço-espectral-temporal otimizada para o dataset MM-Fi:
- Entrada: Tensor CSI de dimensões (B, C, 114, 10) onde:
    - B: Tamanho do lote (batch)
    - C: Canais de entrada (3 para amplitude pura, ou 6 para amplitude + fase sanitizada)
    - 114: Subportadoras OFDM (frequência)
    - 10: Pacotes temporais por quadro (janela de 100 ms)
- Backbone: Extrator convolucional com conexões residuais e normalização em lote.
- Cabeça 1 (Classificação): Mapeia o embedding latente para as 27 ações físicas (Tarefas c e e).
- Cabeça 2 (Pose 3D): Regride as coordenadas (x, y, z) das 17 articulações corporais (Rota B).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    # Permite importação e inspeção de assinaturas antes da instalação do PyTorch
    torch = None
    nn = object
    F = None


class ResidualConvBlock(nn.Module if torch else object):
    """Bloco convolucional residual 2D com Batch Normalization."""

    def __init__(self, in_channels: int, out_channels: int, stride: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class CSISpatialSpectralTemporalNet(nn.Module if torch else object):
    """Rede Neural Convolucional Espaço-Espectral-Temporal com Cabeças Multi-Tarefa.

    Parâmetros:
    -----------
    in_channels : int
        Número de canais de entrada (3 para amplitude, 6 para amplitude + fase).
    num_classes : int
        Número de classes de ação a discriminar (padrão: 27 para o MM-Fi).
    num_joints : int
        Número de articulações do esqueleto (padrão: 17 no Human3.6M).
    latent_dim : int
        Dimensão do vetor de características do backbone (padrão: 256).
    dropout_rate : float
        Taxa de dropout para regularização (padrão: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 27,
        num_joints: int = 17,
        latent_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        if torch is None:
            raise RuntimeError("PyTorch não está instalado no ambiente.")
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        # 1. Camada de Entrada Espectro-Temporal
        # Entrada: (B, C, 114, 10)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 2. Backbone Convolucional Multinível
        # Nível 1: Redução espectral (114 -> 57), mantém tempo (10)
        self.layer1 = ResidualConvBlock(32, 64, stride=(2, 1))
        # Nível 2: Redução espectro-temporal (57 -> 28, 10 -> 5)
        self.layer2 = ResidualConvBlock(64, 128, stride=(2, 2))
        # Nível 3: Redução espectral final (28 -> 14, 5)
        self.layer3 = ResidualConvBlock(128, latent_dim, stride=(2, 1))

        # Pooling Global Adaptativo -> (B, latent_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)

        # 3. Cabeça 1: Classificador de Ações (Tarefas c e e)
        self.head_classificacao = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        # 4. Cabeça 2: Regressor de Pose 3D (Rota B / Reconstrução)
        # Saída: 17 articulações x 3 coordenadas = 51 valores contínuos (metros)
        self.head_pose = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints * 3)
        )

        # Inicializa o bias da cabeça de pose com o prior do centro da cena (Z ~ 3.0 m)
        with torch.no_grad():
            self.head_pose[-1].bias.zero_()
            for j in range(num_joints):
                self.head_pose[-1].bias[j * 3 + 2] = 3.0  # Profundidade média da câmera (Z)

    def forward(
        self,
        x: "torch.Tensor",
        mascara_subportadoras: Optional["torch.Tensor"] = None
    ) -> Dict[str, "torch.Tensor"]:
        """Passo forward da rede.

        Parâmetros:
        -----------
        x : torch.Tensor
            Tensor CSI com dimensões (B, C, 114, 10).
        mascara_subportadoras : torch.Tensor, opcional
            Máscara booleana ou float (114,) para simular ablação de subportadoras (Tarefa e).
        """
        # Se houver máscara de ablação de subportadoras OFDM
        if mascara_subportadoras is not None:
            # mascara formato compatível com (1, 1, 114, 1)
            x = x * mascara_subportadoras.view(1, 1, -1, 1).to(x.device)

        # Passagem pelo backbone
        feat = self.conv_in(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)

        # Vetor latente
        pool = self.global_pool(feat).flatten(1)  # (B, latent_dim)
        embedding = self.dropout(pool)

        # Cabeça 1: Logits de ação (B, num_classes)
        logits_acao = self.head_classificacao(embedding)

        # Cabeça 2: Pose 3D (B, 17, 3)
        pose_3d = self.head_pose(embedding).view(-1, self.num_joints, 3)

        return {
            "logits": logits_acao,
            "pose_3d": pose_3d,
            "embedding": pool
        }


def calcular_mpjpe(
    pose_pred: Union["torch.Tensor", np.ndarray],
    pose_real: Union["torch.Tensor", np.ndarray]
) -> float:
    """Calcula o Mean Per Joint Position Error (MPJPE) em milímetros (mm).

    Parâmetros:
    -----------
    pose_pred : (N, 17, 3) em metros
    pose_real : (N, 17, 3) em metros

    Retorno:
    --------
    float : Erro médio por junta em milímetros.
    """
    if hasattr(pose_pred, "detach"):
        pose_pred = pose_pred.detach().cpu().numpy()
    if hasattr(pose_real, "detach"):
        pose_real = pose_real.detach().cpu().numpy()

    # Distância Euclidiana por junta: sqrt(sum((p - g)^2, axis=-1))
    dists = np.linalg.norm(pose_pred - pose_real, axis=-1)  # shape (N, 17) em metros
    mpjpe_mm = float(np.mean(dists) * 1000.0)  # Converte metros -> milímetros
    return mpjpe_mm
