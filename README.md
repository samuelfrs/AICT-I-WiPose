# Sensoriamento de Poses Humanas Usando Sinais WiFi (Wi-Pose)

Projeto da disciplina **Ações Integradas de Ciência e Tecnologia I (AICT I)**.

## 🎯 Objetivo

Processar sinais de rádio (CSI de WiFi) e treinar um modelo capaz de **classificar
movimentos corporais sem câmera** — a partir apenas de como o corpo humano
deforma o sinal WiFi ao se mover pelo ambiente.

> ### ⚠️ Duas coisas importantes antes de começar
>
> **1. O dataset não é o dos artigos.** Existem duas coisas chamadas "Wi-Pose": o
> artigo do MobiCom 2020 (SUNY Buffalo) e o dataset
> [NjtechCVLab/Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset)
> (Nanjing Tech), que é o que usamos. O dataset foi publicado por outro artigo, o
> **CSI-Former** (Entropy, 2023).
>
> **2. Não dá para fazer pose 3D com este dataset.** As anotações vêm do AlphaPose
> sobre vídeo comum, então são **2D**. E o CSI distribuído traz **apenas a
> amplitude** do sinal, sem a fase. O escopo viável é **classificação de ações** e,
> opcionalmente, regressão de pose 2D.

## 📋 Tarefas do projeto

Conforme o `tema.pdf`:

| | Tarefa |
|---|---|
| **a** | Revisão bibliográfica — Wi-Pose (atividades predefinidas) × Winect (free-form) |
| **b** | Manipulação das estruturas de dados — carregar o CSI e plotar gráficos |
| **c** | Classificação de movimentos — modelo de ML para discriminar as 12 ações |
| **d** | Pré-processamento e calibração — filtragem espacial e temporal |
| **e** | Análise crítica — matrizes de confusão, impacto de subportadoras ruidosas |

## 🚀 Ambiente

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

<sub>Git Bash: `source .venv/Scripts/activate` — PowerShell: `.venv\Scripts\Activate.ps1`</sub>

## 💾 Dataset

[Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset) — 166.600 pacotes
`.mat` com CSI de 5 GHz e anotações de 18 pontos do esqueleto, para 12 ações
executadas por 12 voluntários.

Instruções de download e instalação em [`dataset/README.md`](dataset/README.md).
**Leia esse arquivo antes de escrever código**: o README oficial do dataset
descreve variáveis que não existem no pacote distribuído.

O dataset **não é versionado** (2,4 GB extraído). Cada integrante baixa a própria
cópia.

## 📚 Artigos de Referência

- **Wi-Pose:** Jiang, W., et al. *"Towards 3D human pose construction using WiFi."* MobiCom, 2020.
- **Winect:** Ren, Y., et al. *"Winect: 3D Human Pose Tracking for Free-form Activity Using Commodity WiFi."* IMWUT, 2021.
- **CSI-Former:** Zhou, Y., et al. *"CSI-Former: Pay More Attention to Pose Estimation with WiFi."* Entropy 25(1):20, 2023 — **o artigo que publica o dataset que usamos**.

## 🔍 Como Inspecionar Amostras (Tarefa b)

Para carregar uma amostra `.mat`, extrair o CSI e visualizar graficamente (com janela interativa ou salvando em imagem):

```cmd
# Salva o gráfico em arquivo:
.venv\Scripts\python.exe scripts\inspecionar.py --arquivo "dataset\Wi-Pose\Wi-Pose\Test\bend_009-frame001.mat" --salvar docs/inspecao_amostra.png

# Abre a janela interativa do Matplotlib na tela:
.venv\Scripts\python.exe scripts\inspecionar.py --mostrar
```

## 🗂️ Estrutura

```
AICT-I-WiPose/
├── Artigos/          Artigos de referência em PDF
├── dataset/          Dataset local (ignorado no Git) — ver dataset/README.md
├── docs/             Documentações e figuras geradas (inspecao_amostra.png)
├── scripts/          Scripts executáveis da CLI (inspecionar.py)
├── src/              Módulos de dados e visualização (dados.py, visualizacao.py)
├── tema.pdf          Proposta da disciplina
└── requirements.txt
```
