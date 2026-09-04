"""
Script de linha de comando para inspecionar amostras e gerar gráficos do CSI e pose.
Atende à Tarefa (b) da disciplina AICT I.
"""

import argparse
from pathlib import Path
import sys

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dados import carregar_amostra_mat
from src.visualizacao import plotar_amostra_completa


def main():
    parser = argparse.ArgumentParser(description="Inspecionar e plotar amostra do Wi-Pose")
    parser.add_argument(
        "--arquivo",
        type=str,
        default=r"dataset\Wi-Pose\Wi-Pose\Test\bend_009-frame001.mat",
        help="Caminho para o arquivo .mat"
    )
    parser.add_argument(
        "--salvar",
        type=str,
        default="docs/inspecao_amostra.png",
        help="Caminho onde salvar o gráfico gerado (opcional se usar --mostrar)"
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Abrir a janela interativa do gráfico na tela"
    )
    args = parser.parse_args()

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"Erro: Arquivo não encontrado em '{caminho}'.")
        return

    print(f"--> Carregando arquivo: {caminho}")
    csi, juntas, meta = carregar_amostra_mat(caminho)

    print("\n=== Metadados da Amostra ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n=== Estrutura das Variáveis ===")
    print(f"  CSI: formato {csi.shape} (9 canais x 30 subportadoras x 5 tempos) | Tipo: {csi.dtype}")
    print(f"  Esqueleto: formato {juntas.shape} (18 articulações x [x, y, conf]) | Tipo: {juntas.dtype}")

    print(f"\n--> Gerando visualizações...")
    plotar_amostra_completa(str(caminho), salvar_em=args.salvar, mostrar=args.mostrar)
    print("Concluído!")


if __name__ == "__main__":
    main()
