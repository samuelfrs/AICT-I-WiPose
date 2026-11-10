"""Script de Treinamento e Avaliação da Rede Neural Multi-Head no MM-Fi.

Executa o treinamento do modelo convolucional espaço-espectral-temporal,
calcula as métricas conjuntas de classificação (acurácia, F1, matriz de confusão)
e estimação de pose 3D (MPJPE em mm), e realiza o experimento de ablação
de subportadoras OFDM ruidosas (Tarefas c e e do tema.pdf).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "artifacts" / ".mplcache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Adiciona raiz do repositório ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_rede import MMFiMultiTaskDataset
from src.mmfi import ACOES, CONEXOES_ESQUELETO, NOMES_JUNTAS
from src.rede_neural import CSISpatialSpectralTemporalNet, calcular_mpjpe


def plotar_matriz_confusao(
    cm: np.ndarray,
    classes: list,
    caminho_salvar: Path,
    titulo: str = "Matriz de Confusão — Classificação de Ações (WiFi CSI)",
):
    """Gera e salva o heatmap da matriz de confusão normalizada."""
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_title(titulo, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Classe Prevista", fontsize=11, labelpad=10)
    ax.set_ylabel("Classe Real", fontsize=11, labelpad=10)

    # Imprime os valores nas células
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_salvar, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Matriz de confusao salva em: {caminho_salvar}")


def plotar_ablacao_subportadoras(
    resultados_ablacao: dict,
    caminho_salvar: Path,
):
    """Gera gráfico comparativo do impacto de subportadoras OFDM na acurácia e MPJPE."""
    cenarios = list(resultados_ablacao.keys())
    acuracias = [v["acuracia"] * 100 for v in resultados_ablacao.values()]
    mpjpes = [v["mpjpe_mm"] for v in resultados_ablacao.values()]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)

    color = "tab:blue"
    ax1.set_xlabel("Configuração de Subportadoras OFDM", fontsize=11, labelpad=10)
    ax1.set_ylabel("Acurácia de Classificação (%)", color=color, fontsize=11)
    barras = ax1.bar(
        np.arange(len(cenarios)) - 0.2,
        acuracias,
        width=0.4,
        color=color,
        alpha=0.85,
        label="Acurácia (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 100)

    # Adiciona rótulos sobre as barras
    for b in barras:
        h = b.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("MPJPE de Pose 3D (mm)", color=color, fontsize=11)
    barras2 = ax2.bar(
        np.arange(len(cenarios)) + 0.2,
        mpjpes,
        width=0.4,
        color=color,
        alpha=0.85,
        label="MPJPE (mm)",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    max_m = max(mpjpes) if mpjpes else 300
    ax2.set_ylim(0, max_m * 1.3)

    for b in barras2:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f} mm",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xticks(np.arange(len(cenarios)))
    ax1.set_xticklabels(cenarios, rotation=15, ha="right", fontsize=10)
    plt.title(
        "Análise Crítica: Impacto de Subportadoras OFDM (Tarefa e)",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    fig.tight_layout()
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_salvar, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafico de ablacao salvo em: {caminho_salvar}")


def plotar_comparacao_esqueleto_3d(
    pose_real: np.ndarray,
    pose_pred: np.ndarray,
    caminho_salvar: Path,
    titulo: str = "Pose 3D: Real (Ground Truth) vs Estimada pelo WiFi CSI",
):
    """Plota o esqueleto 3D real vs estimado lado a lado em projeção 3D."""
    fig = plt.figure(figsize=(12, 6), dpi=200)

    for idx, (pose, nome, cor) in enumerate(
        [(pose_real, "Ground Truth (Real)", "tab:green"), (pose_pred, "Predição (WiFi CSI)", "tab:purple")]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        xs = pose[:, 0]
        # Inverte y para manter anatomia humana de pé (pois no MM-Fi y cresce para baixo)
        ys = -pose[:, 1]
        zs = pose[:, 2]

        ax.scatter(xs, zs, ys, color=cor, s=35, label=nome)

        for u, v in CONEXOES_ESQUELETO:
            ax.plot(
                [xs[u], xs[v]],
                [zs[u], zs[v]],
                [ys[u], ys[v]],
                color=cor,
                alpha=0.7,
                linewidth=2,
            )

        ax.set_title(nome, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m — profundidade)")
        ax.set_zlabel("Y invertido (m — altura)")

    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_salvar, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Comparativo de esqueleto 3D salvo em: {caminho_salvar}")


def treinar_modelo(
    split: str = "S2",
    protocolo: str = "P3",
    epocas: int = 15,
    batch_size: int = 16,
    stride: int = 10,
    lr: float = 1e-3,
    lambda_pose: float = 10.0,
    usar_fase: bool = False,
    max_seq_treino: int = None,
    max_seq_teste: int = None,
):
    """Executa o ciclo completo de treinamento, validação e ablação."""
    print("=" * 70, flush=True)
    print(f"[INICIO] TREINAMENTO MULTI-TASK: Split={split}, Protocolo={protocolo}", flush=True)
    print(f"Parametros: epocas={epocas}, batch_size={batch_size}, stride={stride}, usar_fase={usar_fase}", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de execucao: {device}", flush=True)

    # Carrega datasets de treino e teste
    ds_treino = MMFiMultiTaskDataset(
        split=split,
        fase="treino",
        protocolo=protocolo,
        usar_fase=usar_fase,
        stride_quadros=stride,
        max_sequencias=max_seq_treino,
    )

    ds_teste = MMFiMultiTaskDataset(
        split=split,
        fase="teste",
        protocolo=protocolo,
        usar_fase=usar_fase,
        stride_quadros=stride,
        max_sequencias=max_seq_teste,
    )

    print(f"Amostras de treino: {len(ds_treino)} | Amostras de teste: {len(ds_teste)}")
    if len(ds_treino) == 0 or len(ds_teste) == 0:
        raise RuntimeError("Dataset vazio. Verifique o caminho dos dados em dataset/MMFi_leve.")

    dl_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=0)

    in_channels = 6 if usar_fase else 3
    num_classes = len(ds_treino.acoes_lista)
    modelo = CSISpatialSpectralTemporalNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_joints=17,
        latent_dim=256,
        dropout_rate=0.3,
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)
    criterio_cls = nn.CrossEntropyLoss()
    criterio_pose = nn.MSELoss()

    historico = {"loss_treino": [], "acc_teste": [], "mpjpe_teste": []}

    melhor_acc = 0.0
    melhor_mpjpe = float("inf")

    inicio_treino = time.time()

    for epoca in range(1, epocas + 1):
        modelo.train()
        loss_total_acum = 0.0
        n_lotes = 0

        for batch in dl_treino:
            x = batch["csi"].to(device)
            rotulos = batch["rotulo_acao"].to(device)
            poses = batch["pose_3d"].to(device)

            otimizador.zero_grad()
            saida = modelo(x)

            loss_c = criterio_cls(saida["logits"], rotulos)
            loss_p = criterio_pose(saida["pose_3d"], poses)
            loss = loss_c + lambda_pose * loss_p

            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            otimizador.step()

            loss_total_acum += loss.item()
            n_lotes += 1

        scheduler.step()
        loss_media = loss_total_acum / max(1, n_lotes)

        # Avaliação na fase de teste
        modelo.eval()
        corretos = 0
        total_amostras = 0
        soma_mpjpe = 0.0

        with torch.no_grad():
            for batch in dl_teste:
                x = batch["csi"].to(device)
                rotulos = batch["rotulo_acao"].to(device)
                poses = batch["pose_3d"].to(device)

                saida = modelo(x)
                preds = saida["logits"].argmax(dim=1)
                corretos += (preds == rotulos).sum().item()
                total_amostras += rotulos.size(0)

                soma_mpjpe += calcular_mpjpe(saida["pose_3d"], poses) * rotulos.size(0)

        acc_teste = (corretos / total_amostras) if total_amostras > 0 else 0.0
        mpjpe_medio = (soma_mpjpe / total_amostras) if total_amostras > 0 else 0.0

        historico["loss_treino"].append(loss_media)
        historico["acc_teste"].append(acc_teste)
        historico["mpjpe_teste"].append(mpjpe_medio)

        print(
            f"Epoca {epoca:02d}/{epocas:02d} | "
            f"Loss Treino: {loss_media:.4f} | "
            f"Acuracia Teste: {acc_teste * 100:.2f}% | "
            f"MPJPE Teste: {mpjpe_medio:.2f} mm",
            flush=True,
        )

    tempo_total = time.time() - inicio_treino
    print(f"\n[INFO] Treinamento concluido em {tempo_total:.1f} segundos.", flush=True)

    # -----------------------------------------------------------------------
    # Avaliação Final Completa & Matriz de Confusão
    # -----------------------------------------------------------------------
    modelo.eval()
    todas_predicoes = []
    todos_reais = []
    exemplo_pose_real = None
    exemplo_pose_pred = None

    with torch.no_grad():
        for batch in dl_teste:
            x = batch["csi"].to(device)
            rotulos = batch["rotulo_acao"].to(device)
            poses = batch["pose_3d"].to(device)

            saida = modelo(x)
            preds = saida["logits"].argmax(dim=1)

            todas_predicoes.extend(preds.cpu().numpy().tolist())
            todos_reais.extend(rotulos.cpu().numpy().tolist())

            if exemplo_pose_real is None:
                exemplo_pose_real = poses[0].cpu().numpy()
                exemplo_pose_pred = saida["pose_3d"][0].cpu().numpy()

    nomes_acoes_ordenadas = sorted(list(ACOES.keys()))
    cm = confusion_matrix(todos_reais, todas_predicoes, labels=list(range(num_classes)))

    # Salva Matriz de Confusão em PNG
    figura_cm_path = Path("artifacts/figuras/matriz_confusao.png")
    plotar_matriz_confusao(cm, nomes_acoes_ordenadas, figura_cm_path)

    # Salva Comparativo de Pose 3D
    figura_pose_path = Path("artifacts/figuras/pose3d_predicao.png")
    if exemplo_pose_real is not None:
        plotar_comparacao_esqueleto_3d(exemplo_pose_real, exemplo_pose_pred, figura_pose_path)

    # -----------------------------------------------------------------------
    # Experimento de Ablação de Subportadoras OFDM (Tarefa e)
    # -----------------------------------------------------------------------
    print("\n[EXPERIMENTO] Executando Experimento de Ablacao de Subportadoras OFDM (Tarefa e)...")
    cenarios_ablacao = {
        "Todas (114)": None,
        "Sem Bordas (96)": torch.cat(
            [torch.zeros(9), torch.ones(96), torch.zeros(9)]
        ),  # Remove 9 de cada borda
        "Apenas 50% Centrais": torch.cat(
            [torch.zeros(28), torch.ones(58), torch.zeros(28)]
        ),  # 58 centrais
        "Pares Apenas (57)": torch.tensor(
            [1.0 if i % 2 == 0 else 0.0 for i in range(114)]
        ),  # Amostragem intercalada
    }

    resultados_ablacao = {}

    for nome_cenario, mascara in cenarios_ablacao.items():
        corretos_ab = 0
        total_ab = 0
        soma_mpjpe_ab = 0.0

        with torch.no_grad():
            for batch in dl_teste:
                x = batch["csi"].to(device)
                rotulos = batch["rotulo_acao"].to(device)
                poses = batch["pose_3d"].to(device)

                saida = modelo(x, mascara_subportadoras=mascara)
                preds = saida["logits"].argmax(dim=1)

                corretos_ab += (preds == rotulos).sum().item()
                total_ab += rotulos.size(0)
                soma_mpjpe_ab += calcular_mpjpe(saida["pose_3d"], poses) * rotulos.size(0)

        acc_ab = (corretos_ab / total_ab) if total_ab > 0 else 0.0
        mpjpe_ab = (soma_mpjpe_ab / total_ab) if total_ab > 0 else 0.0

        resultados_ablacao[nome_cenario] = {"acuracia": acc_ab, "mpjpe_mm": mpjpe_ab}
        print(f"  - {nome_cenario:20s}: Acuracia = {acc_ab * 100:.2f}% | MPJPE = {mpjpe_ab:.2f} mm")

    figura_ablacao_path = Path("artifacts/figuras/ablacao_subportadoras.png")
    plotar_ablacao_subportadoras(resultados_ablacao, figura_ablacao_path)

    # Salva relatório consolidado de métricas em JSON
    resumo_resultados = {
        "split": split,
        "protocolo": protocolo,
        "epocas": epocas,
        "stride": stride,
        "tempo_segundos": tempo_total,
        "acuracia_final": historico["acc_teste"][-1],
        "mpjpe_final_mm": historico["mpjpe_teste"][-1],
        "historico": historico,
        "resultados_ablacao": resultados_ablacao,
    }

    path_json = Path("artifacts/resultados_rede_neural.json")
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(resumo_resultados, f, indent=2)
    print(f"[OK] Resultados consolidados salvos em: {path_json}")

    return resumo_resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino da Rede Neural Multi-Head WiFi CSI")
    parser.add_argument("--split", type=str, default="S2", choices=["S1", "S2"], help="Split oficial (S1 ou S2)")
    parser.add_argument("--protocolo", type=str, default="P3", choices=["P1", "P2", "P3"], help="Protocolo de ações")
    parser.add_argument("--epocas", type=int, default=12, help="Número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--stride", type=int, default=10, help="Stride de quadros")
    parser.add_argument("--usar-fase", action="store_true", help="Inclui fase sanitizada (6 canais)")
    parser.add_argument("--max-seq-treino", type=int, default=None, help="Limita sequências no treino (debug)")
    parser.add_argument("--max-seq-teste", type=int, default=None, help="Limita sequências no teste (debug)")

    args = parser.parse_args()

    treinar_modelo(
        split=args.split,
        protocolo=args.protocolo,
        epocas=args.epocas,
        batch_size=args.batch_size,
        stride=args.stride,
        usar_fase=args.usar_fase,
        max_seq_treino=args.max_seq_treino,
        max_seq_teste=args.max_seq_teste,
    )
