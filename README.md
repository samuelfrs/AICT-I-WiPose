# Sensoriamento de Poses Humanas Usando Sinais WiFi (Wi-Pose)

Projeto da disciplina **Ações Integradas de Ciência e Tecnologia I (AICT I)**.

## 🎯 Objetivo

Processar sinais de rádio (CSI de WiFi) e treinar um modelo capaz de **classificar
movimentos corporais sem câmera** — a partir apenas de como o corpo humano
deforma o sinal WiFi ao se mover pelo ambiente.

> ### 📌 Trocamos de dataset em 10/09/2026
>
> O projeto começou com o [Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset),
> mas ele só distribui **amplitude** (sem fase) e anotação **2D** — o que deixava
> as tarefas (b) e (d) do enunciado sem objeto.
>
> Agora usamos o **[MM-Fi](https://github.com/ybhbingo/MMFi_dataset)** (NeurIPS 2023),
> que tem **fase do sinal**, **pose 3D**, 114 subportadoras, 27 ações, 40 sujeitos
> e 4 ambientes. O porquê da troca está em [`docs/analise-mmfi.md`](docs/analise-mmfi.md).

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

## 💾 Dataset — MM-Fi

[MM-Fi](https://github.com/ybhbingo/MMFi_dataset) — 1.080 sequências de 297 quadros
(320.760 no total), com CSI de WiFi 5 GHz (amplitude **e fase**, 114 subportadoras,
3 antenas) e pose **3D** de 17 juntas, para 27 ações de 40 voluntários em 4 ambientes.

Só a modalidade `wifi-csi` já são ~16 GB, então o dataset **não é versionado**.
Cada integrante baixa a própria cópia e aponta o caminho:

```cmd
set MMFI_DATASET=C:\caminho\para\MMFi_Dataset
```

<sub>Git Bash: `export MMFI_DATASET=/c/caminho/para/MMFi_Dataset` — PowerShell: `$env:MMFI_DATASET="..."`</sub>

Sem a variável, o código procura em `~/Downloads/datas/MMFi_Dataset`.

O `dataset/` na raiz e o [`dataset/README.md`](dataset/README.md) são do Wi-Pose,
da primeira fase.

## 📖 Documentação

| Documento | Para quê |
| --- | --- |
| [`docs/analise-mmfi.md`](docs/analise-mmfi.md) | **Comece por aqui.** O que o MM-Fi entrega, riscos metodológicos e caminhos possíveis. |
| [`docs/replicar-mmfi.md`](docs/replicar-mmfi.md) | Como replicar o baseline publicado, e o que já vem pronto no toolbox oficial. |
| [`docs/guia-do-projeto.md`](docs/guia-do-projeto.md) | Explicação do projeto do zero (escrito na fase Wi-Pose). |
| [`docs/analise-inicial.md`](docs/analise-inicial.md) | Análise do Wi-Pose — documenta por que trocamos de dataset. |

## 📚 Artigos de Referência

- **MM-Fi:** Yang, J., et al. *"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing."* NeurIPS 2023 — **o dataset que usamos**.
- **Wi-Pose:** Jiang, W., et al. *"Towards 3D human pose construction using WiFi."* MobiCom, 2020.
- **Winect:** Ren, Y., et al. *"Winect: 3D Human Pose Tracking for Free-form Activity Using Commodity WiFi."* IMWUT, 2021.
- **MetaFi++:** Zhou, Y., et al. *"WiFi-Enabled Transformer-Based Human Pose Estimation."* IEEE IoT Journal, 2023 — baseline de WiFi do MM-Fi.

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
