"""Script de Treinamento e Avaliação da Rede Neural Residual Temporal no MM-Fi.

Treina o modelo CSITemporalActionNet sobre janelas temporais de CSI de 3 segundos,
calcula as métricas de classificação por janela e por sequência (Soft Voting),
gera a matriz de confusão e executa o experimento de ablação de subportadoras OFDM (Tarefas c e e).
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
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

# Adiciona raiz do repositório ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_rede import MMFiTemporalDataset
from src.mmfi import ACOES
from src.rede_neural import CSITemporalActionNet


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
    print(f"[OK] Matriz de confusão salva em: {caminho_salvar}")


def plotar_ablacao_subportadoras(
    resultados_ablacao: dict,
    caminho_salvar: Path,
):
    """Gera gráfico comparativo do impacto de subportadoras OFDM na acurácia da rede."""
    cenarios = list(resultados_ablacao.keys())
    acc_seq = [v["acuracia_sequencia"] * 100 for v in resultados_ablacao.values()]
    acc_jan = [v["acuracia_janela"] * 100 for v in resultados_ablacao.values()]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    x = np.arange(len(cenarios))
    largura = 0.35

    barras1 = ax.bar(x - largura / 2, acc_seq, largura, label="Acurácia por Sequência (Soft Voting)", color="tab:blue", alpha=0.9)
    barras2 = ax.bar(x + largura / 2, acc_jan, largura, label="Acurácia por Janela", color="tab:cyan", alpha=0.7)

    ax.set_ylabel("Acurácia Top-1 (%)", fontsize=11)
    ax.set_title("Análise Crítica: Impacto de Subportadoras OFDM na Acurácia (Tarefa e)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cenarios, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, max(max(acc_seq), max(acc_jan)) * 1.35)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for b in barras1:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for b in barras2:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_salvar, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico de ablação salvo em: {caminho_salvar}")


def avaliar_rede(
    modelo: nn.Module,
    dl: DataLoader,
    device: torch.device,
    mascara_subportadoras: Optional[torch.Tensor] = None,
):
    """Calcula acurácia por janela e acurácia agregada por sequência via soft voting."""
    modelo.eval()
    todas_probs = []
    todos_rotulos = []
    todos_seq_ids = []

    with torch.no_grad():
        for batch in dl:
            x = batch["csi"].to(device)
            rotulos = batch["rotulo_acao"]
            seq_ids = batch["seq_idx"]

            saida = modelo(x, mascara_subportadoras=mascara_subportadoras)
            probs = F.softmax(saida["logits"], dim=-1)

            todas_probs.append(probs.cpu().numpy())
            todos_rotulos.append(rotulos.numpy())
            todos_seq_ids.append(seq_ids.numpy())

    todas_probs = np.concatenate(todas_probs, axis=0)
    todos_rotulos = np.concatenate(todos_rotulos, axis=0)
    todos_seq_ids = np.concatenate(todos_seq_ids, axis=0)

    # 1. Métrica por Janela
    preds_janela = todas_probs.argmax(axis=-1)
    acc_janela = float(np.mean(preds_janela == todos_rotulos))

    # 2. Métrica por Sequência (Soft Voting)
    seqs_unicas = np.unique(todos_seq_ids)
    preds_seq = []
    rotulos_seq = []

    for sid in seqs_unicas:
        mask = (todos_seq_ids == sid)
        prob_media_seq = todas_probs[mask].mean(axis=0)
        preds_seq.append(int(prob_media_seq.argmax()))
        rotulos_seq.append(int(todos_rotulos[mask][0]))

    preds_seq = np.array(preds_seq)
    rotulos_seq = np.array(rotulos_seq)
    acc_seq = float(np.mean(preds_seq == rotulos_seq))
    f1_seq = float(f1_score(rotulos_seq, preds_seq, average="macro"))

    return {
        "acc_janela": acc_janela,
        "acc_seq": acc_seq,
        "f1_seq": f1_seq,
        "preds_seq": preds_seq,
        "rotulos_seq": rotulos_seq,
        "preds_janela": preds_janela,
        "rotulos_janela": todos_rotulos,
    }


def treinar_modelo(
    split: str = "S2",
    protocolo: str = "P3",
    epocas: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    seed: int = 0,
    refazer_cache: bool = False,
):
    """Executa o ciclo completo de treinamento, avaliação e ablação."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70, flush=True)
    print(f"[INÍCIO] TREINAMENTO TEMPORAL RESIDUAL: Split={split}, Protocolo={protocolo}", flush=True)
    print(f"Parâmetros: épocas={epocas}, batch_size={batch_size}, lr={lr}, seed={seed}", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de execução: {device}", flush=True)

    # Carrega dataset de treino
    ds_treino = MMFiTemporalDataset(
        split=split,
        fase="treino",
        protocolo=protocolo,
        seed=seed,
        refazer_cache=refazer_cache,
    )

    # Carrega dataset de teste usando os parâmetros do scaler do treino
    ds_teste = MMFiTemporalDataset(
        split=split,
        fase="teste",
        protocolo=protocolo,
        seed=seed,
        media_scaler=ds_treino.media,
        desvio_scaler=ds_treino.desvio,
    )

    num_classes = ds_treino.num_classes
    print(f"Janelas de Treino: {len(ds_treino)} | Janelas de Teste: {len(ds_teste)} | Classes: {num_classes}")

    dl_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=0)

    modelo = CSITemporalActionNet(
        in_features=1368,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout,
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)
    criterio = nn.CrossEntropyLoss()

    historico = {
        "loss_treino": [],
        "acc_janela_teste": [],
        "acc_seq_teste": [],
    }

    inicio_treino = time.time()

    for epoca in range(1, epocas + 1):
        modelo.train()
        loss_total = 0.0
        n_amostras = 0

        for batch in dl_treino:
            x = batch["csi"].to(device)
            y = batch["rotulo_acao"].to(device)

            otimizador.zero_grad()
            saida = modelo(x)
            loss = criterio(saida["logits"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            otimizador.step()

            loss_total += loss.item() * len(y)
            n_amostras += len(y)

        scheduler.step()
        loss_media = loss_total / max(1, n_amostras)

        # Avaliação periódica
        res_teste = avaliar_rede(modelo, dl_teste, device)
        historico["loss_treino"].append(loss_media)
        historico["acc_janela_teste"].append(res_teste["acc_janela"])
        historico["acc_seq_teste"].append(res_teste["acc_seq"])

        if epoca % 5 == 0 or epoca == epocas:
            print(
                f"Época {epoca:02d}/{epocas:02d} | "
                f"Loss Treino: {loss_media:.4f} | "
                f"Acc Janela Teste: {res_teste['acc_janela'] * 100:.2f}% | "
                f"Acc Seq Teste: {res_teste['acc_seq'] * 100:.2f}% (F1: {res_teste['f1_seq']:.3f})",
                flush=True,
            )

    tempo_total = time.time() - inicio_treino
    print(f"\n[INFO] Treinamento concluído em {tempo_total:.1f} segundos.", flush=True)

    # -----------------------------------------------------------------------
    # Avaliação Final Completa & Matriz de Confusão (Tarefa e)
    # -----------------------------------------------------------------------
    res_final = avaliar_rede(modelo, dl_teste, device)
    nomes_classes = ds_treino.acoes_lista
    cm_seq = confusion_matrix(res_final["rotulos_seq"], res_final["preds_seq"], labels=list(range(num_classes)))

    figura_cm_path = Path("artifacts/figuras/matriz_confusao.png")
    plotar_matriz_confusao(
        cm_seq,
        [f"{a}: {ACOES.get(a, '')[:15]}" for a in nomes_classes],
        figura_cm_path,
        titulo=f"Matriz de Confusão — Soft Voting por Sequência ({split} / {protocolo})",
    )

    # -----------------------------------------------------------------------
    # Experimento de Ablação de Subportadoras OFDM (Tarefa e)
    # -----------------------------------------------------------------------
    print("\n[EXPERIMENTO] Executando Experimento de Ablação de Subportadoras OFDM (Tarefa e)...")
    cenarios_ablacao = {
        "Todas (114)": None,
        "Sem Bordas (96)": torch.cat(
            [torch.zeros(9), torch.ones(96), torch.zeros(9)]
        ),
        "Apenas 50% Centrais": torch.cat(
            [torch.zeros(28), torch.ones(58), torch.zeros(28)]
        ),
        "Pares Apenas (57)": torch.tensor(
            [1.0 if i % 2 == 0 else 0.0 for i in range(114)]
        ),
    }

    resultados_ablacao = {}
    for nome_cenario, mascara in cenarios_ablacao.items():
        res_ab = avaliar_rede(modelo, dl_teste, device, mascara_subportadoras=mascara)
        resultados_ablacao[nome_cenario] = {
            "acuracia_sequencia": res_ab["acc_seq"],
            "acuracia_janela": res_ab["acc_janela"],
            "f1_sequencia": res_ab["f1_seq"],
        }
        print(
            f"  - {nome_cenario:20s}: Acc Seq = {res_ab['acc_seq'] * 100:.2f}% | "
            f"Acc Janela = {res_ab['acc_janela'] * 100:.2f}%"
        )

    figura_ablacao_path = Path("artifacts/figuras/ablacao_subportadoras.png")
    plotar_ablacao_subportadoras(resultados_ablacao, figura_ablacao_path)

    # Relatório de classificação detalhado
    report_dict = classification_report(
        res_final["rotulos_seq"],
        res_final["preds_seq"],
        target_names=nomes_classes,
        output_dict=True,
        zero_division=0,
    )

    acaso = 1.0 / num_classes
    resumo_resultados = {
        "modelo": "CSITemporalActionNet",
        "split": split,
        "protocolo": protocolo,
        "epocas": epocas,
        "tempo_segundos": tempo_total,
        "acaso": acaso,
        "acuracia_final_sequencia": res_final["acc_seq"],
        "acuracia_final_janela": res_final["acc_janela"],
        "f1_macro_sequencia": res_final["f1_seq"],
        "matriz_confusao": cm_seq.tolist(),
        "relatorio_classificacao": report_dict,
        "resultados_ablacao": resultados_ablacao,
        "historico": historico,
    }

    path_json = Path("artifacts/resultados_rede_neural.json")
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(resumo_resultados, f, indent=2)
    print(f"[OK] Resultados consolidados salvos em: {path_json}")

    return resumo_resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino da Rede Neural Residual WiFi CSI")
    parser.add_argument("--split", type=str, default="S2", choices=["S1", "S2"], help="Split oficial (S1 ou S2)")
    parser.add_argument("--protocolo", type=str, default="P3", choices=["P1", "P2", "P3"], help="Protocolo de ações")
    parser.add_argument("--epocas", type=int, default=25, help="Número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado")
    parser.add_argument("--dropout", type=float, default=0.3, help="Taxa de dropout")
    parser.add_argument("--seed", type=int, default=0, help="Semente pseudoaleatória")
    parser.add_argument("--refazer-cache", action="store_true", help="Reconstrói cache das janelas")

    args = parser.parse_args()

    treinar_modelo(
        split=args.split,
        protocolo=args.protocolo,
        epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        refazer_cache=args.refazer_cache,
    )
