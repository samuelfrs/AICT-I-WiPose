"""
Módulo de visualização para inspecionar o CSI (antenas, subportadoras)
e esqueletos 2D do dataset Wi-Pose. Atende à Tarefa (b) da disciplina.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from src.dados import CONEXOES_ESQUELETO, NOMES_JUNTAS, carregar_amostra_mat


def plotar_amostra_completa(
    caminho_mat: str,
    salvar_em: Optional[str] = None,
    mostrar: bool = True
):
    """
    Gera um painel com:
    1. Resposta em frequência do CSI (amplitude por subportadora nas 9 combinações de antena).
    2. Evolução temporal dos 5 pacotes para as subportadoras centrais.
    3. Mapa de calor (Heatmap) do CSI (9 canais x 30 subportadoras).
    4. Esqueleto 2D reconstruído (18 articulações) com conexões anatômicas.
    """
    csi, juntas, meta = carregar_amostra_mat(caminho_mat)
    # csi: (9 canais, 30 subportadoras, 5 pacotes no tempo)
    # juntas: (18 juntas, 3 [x, y, conf])

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Ação: {meta['acao'].upper()} | Ensaio: {meta['ensaio']} | Frame: {meta['frame']}",
        fontsize=16,
        fontweight="bold"
    )

    # 1. Amplitude por subportadora (no último instante temporal t=4)
    ax1 = fig.add_subplot(2, 2, 1)
    for canal in range(9):
        ax1.plot(csi[canal, :, -1], label=f"Canal {canal+1}", alpha=0.7)
    ax1.set_title("Amplitude por Subportadora OFDM (9 Canais Tx-Rx)", fontsize=12)
    ax1.set_xlabel("Índice da Subportadora (0 a 29)")
    ax1.set_ylabel("Amplitude do CSI")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Mapa de Calor (Média temporal de amplitude por canal e subportadora)
    ax2 = fig.add_subplot(2, 2, 2)
    csi_medio = csi.mean(axis=-1)  # (9, 30)
    im = ax2.imshow(csi_medio, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax2, label="Amplitude Média")
    ax2.set_title("Mapa de Calor do CSI (9 Canais x 30 Subportadoras)", fontsize=12)
    ax2.set_xlabel("Subportadora")
    ax2.set_ylabel("Par de Antena (0-8)")

    # 3. Variação nos 5 instantes de tempo
    ax3 = fig.add_subplot(2, 2, 3)
    tempo = np.arange(5)
    # Mostra subportadora central (15) para cada canal
    for canal in range(9):
        ax3.plot(tempo, csi[canal, 15, :], marker="o", label=f"Ch {canal+1}")
    ax3.set_title("Dinâmica Temporal (Subportadora Central #15)", fontsize=12)
    ax3.set_xlabel("Pacote / Tempo discreto")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Esqueleto 2D (AlphaPose)
    ax4 = fig.add_subplot(2, 2, 4)
    x = juntas[:, 0]
    y = juntas[:, 1]
    conf = juntas[:, 2]

    # Desenha as conexões entre articulações se confiança for suficiente
    for i, j in CONEXOES_ESQUELETO:
        if conf[i] > 0.05 and conf[j] > 0.05:
            ax4.plot([x[i], x[j]], [y[i], y[j]], color="teal", linewidth=2.5, zorder=1)

    # Plota pontos das juntas
    scatter = ax4.scatter(x, y, c=conf, cmap="plasma", s=60, zorder=2, vmin=0, vmax=1)
    fig.colorbar(scatter, ax=ax4, label="Confiança (AlphaPose)")

    # Inverte eixo Y para corresponder ao sistema de coordenadas de imagem (origem no topo)
    ax4.invert_yaxis()
    ax4.set_title("Esqueleto 2D (18 Juntas - AlphaPose)", fontsize=12)
    ax4.set_xlabel("Pixel X")
    ax4.set_ylabel("Pixel Y")
    ax4.set_aspect("equal", adjustable="datalim")
    ax4.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()

    if salvar_em:
        Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso em: {salvar_em}")

    if mostrar:
        plt.show()
    else:
        plt.close()
