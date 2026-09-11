"""
Módulo de visualização do MM-Fi: CSI (amplitude e fase) e esqueleto 3D.
Atende à Tarefa (b) da disciplina, agora com as duas componentes do sinal.

Segue o mesmo formato de painel do src/visualizacao.py (Wi-Pose), adaptado para
o que o MM-Fi entrega a mais: fase do sinal e pose tridimensional.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.mmfi import (
    CONEXOES_ESQUELETO,
    FRONTEIRA_20MHZ,
    N_QUADROS,
    carregar_pose_sequencia,
    carregar_quadro,
    sanitizar_fase,
)


def _desenhar_esqueleto_3d(ax, juntas: np.ndarray, titulo: str) -> None:
    """Desenha o esqueleto de 17 juntas num eixo 3D do matplotlib.

    ⚠️ O eixo y do MM-Fi cresce para BAIXO (cabeça em y negativo). Usamos -y
    como altura para a pessoa aparecer em pé.
    """
    x, y, z = juntas[:, 0], juntas[:, 1], juntas[:, 2]

    for i, j in CONEXOES_ESQUELETO:
        ax.plot([x[i], x[j]], [z[i], z[j]], [-y[i], -y[j]],
                color="teal", linewidth=2.0, zorder=1)

    ax.scatter(x, z, -y, c="crimson", s=25, zorder=2, depthshade=False)

    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Z (m)", labelpad=2)
    ax.set_zlabel("Altura (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_title(titulo, fontsize=12)

    # Escala igual nos três eixos, senão o corpo aparece distorcido. Limites
    # iguais não bastam no mplot3d: sem set_box_aspect a caixa continua achatada.
    pontos = np.stack([x, z, -y])
    centro = pontos.mean(axis=1)
    alcance = float(np.max(np.ptp(pontos, axis=1))) / 2 or 0.5
    for definir, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centro):
        definir(c - alcance, c + alcance)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-75)


def plotar_amostra_completa(
    sujeito: str = "S01",
    acao: str = "A01",
    quadro: int = 1,
    raiz: Optional[str] = None,
    salvar_em: Optional[str] = None,
    mostrar: bool = False,
):
    """
    Gera um painel com:
    1. Amplitude do CSI por subportadora, nas 3 antenas.
    2. Fase crua vs. fase sanitizada — mostra o offset linear que a tarefa (d) remove.
    3. Mapa de calor da amplitude (subportadoras x tempo).
    4. Esqueleto 3D de 17 juntas (Human3.6M).

    Lê apenas um arquivo .mat (~53 KB) e um ground_truth.npy (~60 KB).
    """
    amplitude, fase, juntas, meta = carregar_quadro(sujeito, acao, quadro, raiz)
    fase_saneada = sanitizar_fase(fase)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"{meta['sujeito']} · {meta['acao']} ({meta['nome_acao']}) · "
        f"quadro {meta['quadro']} · ambiente {meta['ambiente']} · {meta['protocolo']}",
        fontsize=14, fontweight="bold",
    )

    subportadoras = np.arange(amplitude.shape[1])
    instante = 0  # primeira das 10 amostras temporais do quadro

    # --- 1. Amplitude por subportadora -------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    for antena in range(amplitude.shape[0]):
        ax1.plot(subportadoras, amplitude[antena, :, instante],
                 label=f"Antena RX {antena + 1}", alpha=0.85, linewidth=1.3)
    ax1.set_title("Amplitude por subportadora OFDM (114 subportadoras)", fontsize=12)
    ax1.set_xlabel("Índice da subportadora (0 a 113)")
    ax1.set_ylabel("Amplitude do CSI (dB)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    if meta["valores_corrigidos"]:
        ax1.text(0.02, 0.03,
                 f"{meta['valores_corrigidos']} valor(es) não-finito(s) corrigido(s)",
                 transform=ax1.transAxes, fontsize=8, color="darkred")

    # --- 2. Fase: crua, unwrap e sanitizada --------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(subportadoras, fase[0, :, instante], color="tab:gray",
             alpha=0.9, linewidth=0.9, linestyle=":", label="crua (envolvida em ±π)")
    ax2.plot(subportadoras, np.unwrap(fase[0, :, instante]), color="tab:orange",
             linewidth=1.3, label="após unwrap")
    ax2.plot(subportadoras, fase_saneada[0, :, instante], color="tab:blue",
             linewidth=1.7, label="sanitizada (rampa removida por bloco)")

    # A fronteira entre os dois blocos de 20 MHz: o salto ali é estrutural,
    # não ruído, e ajustar uma reta só sobre as 114 o ignora.
    ax2.axvline(FRONTEIRA_20MHZ - 0.5, color="crimson", linestyle="--",
                linewidth=1.2, alpha=0.8)
    ax2.text(FRONTEIRA_20MHZ + 1, ax2.get_ylim()[1], " fronteira 20 MHz ",
             color="crimson", fontsize=8, va="top")
    ax2.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax2.set_title("Fase do CSI — antena RX 1 (tarefa d)", fontsize=12)
    ax2.set_xlabel("Índice da subportadora")
    ax2.set_ylabel("Fase (rad)")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- 3. Mapa de calor: subportadora x tempo ----------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    imagem = ax3.imshow(amplitude[0], aspect="auto", cmap="viridis",
                        origin="lower", interpolation="nearest")
    fig.colorbar(imagem, ax=ax3, label="Amplitude (dB)")
    ax3.set_title("Amplitude na janela de 100 ms — antena RX 1", fontsize=12)
    ax3.set_xlabel("Amostra temporal (10 em 100 ms)")
    ax3.set_ylabel("Subportadora")

    # --- 4. Esqueleto 3D ---------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _desenhar_esqueleto_3d(ax4, juntas, "Esqueleto 3D (17 juntas, Human3.6M)")

    fig.tight_layout()
    _finalizar(fig, salvar_em, mostrar)
    return fig


def plotar_movimento(
    sujeito: str = "S01",
    acao: str = "A01",
    quadros=(1, 75, 150, 225, 297),
    raiz: Optional[str] = None,
    salvar_em: Optional[str] = None,
    mostrar: bool = False,
):
    """Esqueleto 3D em vários instantes da mesma sequência, lado a lado.

    Mostra o movimento acontecendo. Custa quase nada: lê um único .npy com a
    sequência inteira, sem tocar em nenhum arquivo de CSI.
    """
    poses = carregar_pose_sequencia(sujeito, acao, raiz)

    fora = [q for q in quadros if not 1 <= q <= N_QUADROS]
    if fora:
        raise ValueError(
            f"Quadros fora do intervalo 1..{N_QUADROS}: {fora}. "
            "Índice negativo daria a pose errada em silêncio."
        )

    fig = plt.figure(figsize=(4 * len(quadros), 4.5))
    for posicao, quadro in enumerate(quadros, start=1):
        ax = fig.add_subplot(1, len(quadros), posicao, projection="3d")
        _desenhar_esqueleto_3d(ax, poses[quadro - 1], f"quadro {quadro}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    from src.mmfi import ACOES
    fig.suptitle(f"{sujeito} · {acao} ({ACOES[acao]}) — movimento ao longo da sequência",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _finalizar(fig, salvar_em, mostrar)
    return fig


def _finalizar(fig, salvar_em: Optional[str], mostrar: bool) -> None:
    if salvar_em:
        Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {salvar_em}")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
