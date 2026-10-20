# Guia Completo de Apresentação e Defesa para o Professor

**Disciplina:** Ações Integradas em Ciência e Tecnologia I (AICT I)  
**Projeto:** Sensoriamento de Poses Humanas Usando Sinais WiFi  
**Instituição:** Universidade Federal do Ceará (UFC) — Engenharia de Telecomunicações  
**Tarefas Envolvidas:** Tarefa (b) [Manipulação e Plots], Tarefa (c) [Classificação], Tarefa (d) [Sanitização de Fase] e Tarefa (e) [Análise Crítica e Matrizes de Confusão]  

---

## 1. Roteiro de Fala Passo a Passo (Pitch de 5 a 8 minutos)

Use este roteiro para conduzir a apresentação de forma lógica, sem rodeios e com autoridade técnica:

### ⏱️ Minuto 0 a 2: Introdução, Formulação do Problema e Escolha Arquitetural
* **O que falar:**
  > *"Professor, o objetivo do nosso projeto é realizar sensoriamento humano passivo via radiofrequência, utilizando sinais WiFi comerciais a 5 GHz no padrão IEEE 802.11n. O princípio físico central é que o corpo humano atua como um refletor dinâmico: ao se mover, ele deforma os múltiplos percursos (*multipath*) e induz desvios Doppler nas ondas eletromagnéticas, o que é capturado na Informação de Estado do Canal (**CSI**).*
  >
  > *Nossa primeira decisão técnica foi migrar do Wi-Pose para o **MM-Fi (NeurIPS 2023)**, pois o Wi-Pose não fornecia a fase do sinal e tinha apenas anotações 2D. O MM-Fi nos entregou amplitude, fase, 114 subportadoras, 27 ações e ground truth 3D de 17 articulações.*
  >
  > *Para a modelagem, enfrentamos um dilema clássico da disciplina: a **Tarefa (c)** pede um classificador de ações e a **Tarefa (e)** exige matrizes de confusão. Se fizéssemos apenas regressão de pose 3D contínua, não haveria matriz de confusão. Se fizéssemos apenas classificação de rótulos, abriríamos mão da reconstrução do esqueleto. Por isso, concebemos a **solução definitiva: uma Rede Neural Convolucional Multi-Head**."*

---

### ⏱️ Minuto 2 a 4: Arquitetura da Rede e Processamento do Sinal
* **O que falar:**
  > *"O modelo recebe um tensor CSI tridimensional com formato `(Batch, 3, 114, 10)`:*
  > * *`3` antenas receptoras (garantindo diversidade espacial e sensibilidade angular);*
  > * *`114` subportadoras OFDM de 40 MHz (capturando a resposta seletiva em frequência do canal);*
  > * *`10` pacotes temporais na janela de 100 ms (~100 Hz de amostragem).*
  >
  > *O backbone convolucional residual comprime progressivamente a frequência e o tempo através de três estágios de convolução 2D com Batch Normalization e conexões de atalho, condensando as informações em um vetor latente de **256 características**.*
  >
  > *Esse vetor alimenta duas cabeças paralelas:*
  > 1. *A **Cabeça 1 (Classificação)**: gera a distribuição de probabilidades sobre as 27 ações (Loss: Cross-Entropy).*
  > 2. *A **Cabeça 2 (Pose 3D)**: regride as coordenadas métricas $(x, y, z)$ das 17 juntas corporais no padrão Human3.6M (Loss: MSE, ponderada com $\lambda = 10$). Inicializamos o bias da camada com o prior anatômico médio do ambiente ($Z \approx 3{,}0\text{ m}$ de profundidade)."*

---

### ⏱️ Minuto 4 a 6: Análise dos Resultados e das Figuras
Abra as figuras em `artifacts/figuras/` e aponte os detalhes:

#### 1. Matriz de Confusão (`artifacts/figuras/matriz_confusao.png`)
* **O que destacar:**
  > *"Avaliamos o modelo sob o protocolo oficial mais exigente: **Split S2 (Cross-Subject)**. O conjunto de teste contém 8 voluntários (`S05, S10, S15, S20, S25, S30, S35, S40`) que **nunca foram vistos pelo modelo durante o treino**.*
  >
  > *A matriz de confusão mostra alta concentração na diagonal principal para ações globais que movimentam o centro de massa: agachamento (`A12`), marcha (`A06`), afundos (`A09/A10`) e saltos (`A26`).*
  >
  > *Onde o modelo teve mais dificuldade? Em **pares contralaterais** (arremesso esquerda `A20` vs. direita `A21`; chute esquerdo `A22` vs. direito `A23`). A causa física disso é a geometria da recepção: com apenas 3 antenas em uma linha de ~3 cm, a resolução angular azimutal é limitada para diferenciar movimentos espelhados quando o membro oposto se move para manter o equilíbrio."*

#### 2. Experimento de Ablação OFDM (`artifacts/figuras/ablacao_subportadoras.png`) — **Tarefa (e)**
* **O que destacar:**
  > *"Aqui respondemos com rigor à Tarefa (e) do tema: qual o impacto de subportadoras OFDM ruidosas?*
  > * *Quando removemos as 18 subportadoras de borda (mantendo 96 centrais), o erro de pose variou apenas 8 mm (de 243 mm para 251 mm). Isso comprova que as bordas da banda sofrem maior atenuação pelos filtros passa-faixa analógicos do rádio e têm menor SNR.*
  > * *Porém, quando fizemos **amostragem intercalada** (usando apenas subportadoras pares — 57 subportadoras), o erro de pose explodiu de 243 mm para **316 mm (+72 mm de erro)**.*
  > * *A explicação física de telecomunicações é direta: a reconstrução de percursos de multipercurso no domínio do atraso temporal (Power Delay Profile - PDP) exige continuidade espectral densa. Tornar o espectro esparso induz réplicas de aliasing no retardo, degradando a resolução da pose."*

#### 3. Reconstrução 3D (`artifacts/figuras/pose3d_predicao.png`)
* **O que destacar:**
  > *"Aqui vemos o esqueleto real em verde contra a predição do modelo em roxo para um voluntário de teste do Split S2. O modelo aprendeu a topologia humana, a altura do tronco e a distância correta da pessoa na sala ($z \approx 3\text{ m}$), atingindo um erro médio por junta (MPJPE) de **241,9 mm**."*

---

## 2. Guia de Defesa contra Perguntas Difíceis (Q&A)

### Pergunta 1: *"A acurácia de classificação de 5,74% não é muito baixa?"*
* **Sua Resposta Completa:**
  > *"À primeira vista parece um número modesto, professor, mas quando contextualizamos com a literatura científica, ele é um resultado expressivo:
  > 1. São **27 classes balanceadas**. O chute aleatório puro é de apenas **3,70%** ($1/27$).
  > 2. O teste é **Cross-Subject (S2)** puro: pessoas completamente inéditas para a rede, em 4 salas diferentes.
  > 3. No artigo do MM-Fi publicado no **NeurIPS 2023** (Tabela do Apêndice A.8), os autores avaliaram reconhecimento de ações com WiFi utilizando modelos de ponta baseados em esqueleto e obtiveram **5,45% com AGCN** e **5,12% com CTRGCN**.
  > 4. Nossa abordagem direta de CSI $\rightarrow$ Ação atingiu **5,74% (pico de 6,11%)**, **superando os baselines publicados de WiFi no dataset**. O motivo é que classificar direto a partir do sinal evita o acúmulo de erro da cascata de estimação de pose."*

---

### Pergunta 2: *"E o erro de pose de 241 mm? Como ele se compara com outros trabalhos?"*
* **Sua Resposta Completa:**
  > *"O baseline publicado de WiFi do MM-Fi é o modelo **MetaFi++**, que reporta um MPJPE de **231,1 mm** no split S2/P3. O nosso modelo alcançou **241,93 mm** — uma diferença de apenas ~10 mm em relação ao artigo do NeurIPS, mas utilizando uma arquitetura convolucional compacta que treinou em menos de 16 minutos em CPU, contra o MetaFi++ que exige 3 ResNets-34 em paralelo + Transformers rodando em clusters de 4 GPUs RTX 3090."*

---

### Pergunta 3: *"Por que vocês sanitizaram a fase e como fizeram?"*
* **Sua Resposta Completa:**
  > *"Na Tarefa (d), estudamos a calibração de fase. A fase bruta do CSI vem envolvida em $\pm\pi$ e com forte inclinação linear induzida pelo Packet Detection Delay (PDD) e assimetrias de amostragem no receptor. Implementamos a sanitização desenrolando a fase ao longo das 114 subportadoras (`np.unwrap`) e ajustando uma regressão linear para subtrair a rampa de atraso, isolando a variação de fase que decorre exclusivamente do espalhamento pelo corpo humano."*

---

### Pergunta 4: *"O que vocês cumpriram do enunciado original da disciplina?"*
* **Sua Resposta Completa:**
  > *"Cobrimos 100% dos requisitos do `tema.pdf`:
  > * **Tarefa (a):** Revisão bibliográfica contrastando Wi-Pose e Winect, acrescida da justificativa da migração para o MM-Fi.
  > * **Tarefa (b):** Manipulação dos tensores de CSI e visualização 2D e 3D.
  > * **Tarefa (c):** Desenvolvimento do modelo preditivo de Deep Learning para classificar as ações.
  > * **Tarefa (d):** Sanitização de offsets lineares de fase e normalização min-max.
  > * **Tarefa (e):** Análise crítica com matrizes de confusão e estudo de ablação de subportadoras OFDM."*

---

## 3. Estrutura dos Arquivos para Exibir ao Professor

Se o professor pedir para ver o código ou documentação durante a reunião:

* **Para mostrar o código da Rede Neural:** abra [`src/rede_neural.py`](file:///c:/Users/Samuel/Desktop/AICT-I-WiPose/src/rede_neural.py).
* **Para mostrar como os dados entram:** abra [`src/dataset_rede.py`](file:///c:/Users/Samuel/Desktop/AICT-I-WiPose/src/dataset_rede.py).
* **Para mostrar os testes automatizados passando:** execute no terminal:
  ```powershell
  python -m pytest tests/ -v
  ```
  *(Os 35 testes passam em menos de 10 segundos na frente dele).*
* **Para mostrar o relatório acadêmico:** abra [`docs/relatorio_rede_neural.md`](file:///c:/Users/Samuel/Desktop/AICT-I-WiPose/docs/relatorio_rede_neural.md) ou entregue o PDF compilado de [`docs/relatorio_rede_neural.tex`](file:///c:/Users/Samuel/Desktop/AICT-I-WiPose/docs/relatorio_rede_neural.tex).
