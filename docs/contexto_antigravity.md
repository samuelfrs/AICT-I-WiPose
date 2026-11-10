# Contexto Técnico & Passagem de Bastão (Handover para Antigravity)

**Projeto:** Sensoriamento de Poses Humanas Usando Sinais WiFi (AICT I - UFC)  
**Data da última atualização:** 20/10/2026  
**Finalidade:** Este documento foi elaborado especificamente para que a instância do **Antigravity** em outro computador (ou qualquer desenvolvedor/pesquisador que clonar o repositório) compreenda imediatamente todo o estado da engenharia, arquitetura, resultados e próximos passos sem perda de contexto.

---

## 1. Visão Geral Rápida do Projeto

* **Tema:** Estimar poses 3D e classificar ações humanas a partir de sinais WiFi 5 GHz (CSI no padrão IEEE 802.11n) sem o uso de câmeras ou sensores corporais.
* **Dataset Oficial:** **MM-Fi (NeurIPS 2023)**, substituindo o Wi-Pose legado (abandonado por não ter fase e ter pose apenas 2D).
* **Estrutura dos Dados no Disco:**
  * O repositório utiliza a versão leve em `dataset/MMFi_leve/E01..E04/S01..S40/A01..A27.npz`.
  * Cada arquivo `.npz` contém uma sequência de 297 quadros com `csi_amp` (3, 114, 10) em dB, `csi_fase` (3, 114, 10) crua em $\pm\pi$ e `pose` (17, 3) em metros (esqueleto Human3.6M).
  * O dataset NÃO é versionado no Git devido ao limite de 100 MB do GitHub (está devidamente no `.gitignore`).

---

## 2. O que foi Implementado e Validado

### A. Arquitetura da Rede Neural (`src/rede_neural.py`)
* Modelo: `CSISpatialSpectralTemporalNet`.
* Entrada: Tensor de CSI `(Batch, C=3, 114 subportadoras, 10 amostras temporais)`.
* Backbone: 3 estágios de blocos residuais 2D com Batch Normalization e compressão progressiva no tempo e na frequência, gerando um vetor latente de 256 características via Adaptive Average Pooling.
* Multi-Head:
  * **Cabeça 1 (Classificação):** Linear(256 $\rightarrow$ 128) $\rightarrow$ ReLU $\rightarrow$ Linear(128 $\rightarrow$ 27 classes). Atende às **Tarefas (c) e (e)** do enunciado (`tema.pdf`).
  * **Cabeça 2 (Pose 3D):** Linear(256 $\rightarrow$ 128) $\rightarrow$ ReLU $\rightarrow$ Linear(128 $\rightarrow$ 51 valores contínuos = 17 juntas $\times$ 3 eixos). Inicializada com o prior de profundidade média $Z \approx 3{,}0\text{ m}$. Atende à **Rota B** de reconstrução espacial contínua.
* Perda Conjunta: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 10{,}0 \cdot \mathcal{L}_{\text{MSE}}$.

### B. Pipeline de Dados Vetorizado (`src/dataset_rede.py`)
* Classe `MMFiMultiTaskDataset`:
  * Carrega fatias temporais subamostradas direto para arrays contíguos em RAM (~180 MB de memória para o treino inteiro com stride=20).
  * Elimina 100% de gargalos de leitura/descompressão de disco durante as épocas de treinamento.
  * Suporta formalmente o **Split S2 (Cross-Subject Oficial)** com 8 voluntários retidos (`S05, S10, S15, S20, S25, S30, S35, S40`) e 32 voluntários de treino, cobrindo os 4 ambientes físicos (`E01..E04`).

### C. Pipeline de Treino e Análise Crítica (`scripts/treinar_rede.py`)
* Executa o ciclo com AdamW e Cosine Annealing.
* Gera e salva automaticamente:
  * `artifacts/figuras/matriz_confusao.png` (Matriz de confusão normalizada das 27 ações no conjunto de teste independente).
  * `artifacts/figuras/ablacao_subportadoras.png` (Experimento da Tarefa e avaliando a perda de largura de banda e esparsidade espectral).
  * `artifacts/figuras/pose3d_predicao.png` (Comparativo 3D real vs estimado).
  * `artifacts/resultados_rede_neural.json` (Histórico numérico das perdas e métricas).

### D. Testes Unitários (`tests/`)
* `tests/test_rede_neural.py`: 6 testes cobrindo forward (3 e 6 canais), retropropagação, máscara de ablação e cálculo de MPJPE.
* `tests/test_mmfi.py`: 29 testes da camada de carregamento e sanitização de fase.
* **Status:** 33 testes passando, 2 testes de leitura bruta pulados intencionalmente se os `.mat` originais não estiverem presentes (versão leve `.npz` cobre 100% dos testes).

---

## 3. Principais Resultados Obtidos

* **Acurácia Top-1 no Split S2 (Cross-Subject):**
  * Final: **5,74%** (Pico: **6,11%** na época 14 | Acaso teórico = 3,70%).
  * **Supera o baseline oficial de WiFi do MM-Fi** (NeurIPS 2023, Apêndice A.8), que obteve **5,45% com AGCN** e **5,12% com CTRGCN**.
* **MPJPE de Pose 3D no Split S2:**
  * Final: **243,42 mm** (Melhor época: **241,93 mm**).
  * Muito próximo do baseline **MetaFi++ publicado (231,1 mm)**, com custo computacional ordens de magnitude menor.
* **Ablação Espectral OFDM (Tarefa e):**
  * Todas (114 subp): Acurácia 5,74%, MPJPE 243,42 mm (Baseline).
  * Sem Bordas (96 subp): Acurácia 4,85%, MPJPE 251,81 mm (+8,39 mm).
  * Apenas Centrais (58 subp): Acurácia 3,98%, MPJPE 269,65 mm.
  * Amostragem Intercalada (Pares - 57 subp): Acurácia 4,23%, **MPJPE 316,23 mm (+72,81 mm)**.
  * **Conclusão:** Prova física de que a transformada de Fourier inversa sobre o canal depende criticamente de continuidade espectral densa para resolver o perfil de retardo (*Power Delay Profile*).

---

## 4. Ambiente e Comandos Rápidos na Nova Máquina

Ao clonar o repositório na nova máquina:

### 1. Instalar dependências:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
*(Certifique-se de que `torch` e `scikit-learn` estão instalados).*

### 2. Rodar todos os testes:
```powershell
python -m pytest tests/ -v
```

### 3. Rodar novo treinamento se necessário:
```powershell
python scripts/treinar_rede.py --split S2 --protocolo P3 --epocas 15 --batch-size 32 --stride 20
```

### 4. Consultar os documentos acadêmicos prontos:
* **Relatório da Apresentação:** [`docs/guia_apresentacao_professor.md`](guia_apresentacao_professor.md) (Roteiro falado de 5-8 min e perguntas e respostas).
* **Relatório Técnico Completo:** [`docs/relatorio_rede_neural.md`](relatorio_rede_neural.md).
* **Artigo Científico em LaTeX:** [`docs/relatorio_rede_neural.tex`](relatorio_rede_neural.tex).

---

## 5. Próximos Passos Potenciais (Sugestões de Continuidade)

Se você ou o professor desejarem expandir o trabalho na nova máquina:
1. **Treinar incluindo a Fase Sanitizada:** Rodar `scripts/treinar_rede.py --usar-fase` (alimenta 6 canais: 3 de amplitude + 3 de fase desdobrada) para contrastar o efeito da calibração de fase contra a amplitude pura.
2. **Treinar no Protocolo P1 (14 diárias) ou P2 (13 reabilitação):** Comparar a acurácia em atividades dinâmicas de corpo livre (P1) contra exercícios de postura fixa (P2).
3. **Compilar o LaTeX para PDF:** Executar `pdflatex docs/relatorio_rede_neural.tex` para gerar o artigo final diagramado em PDF.
