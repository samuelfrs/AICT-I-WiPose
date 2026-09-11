# Análise Inicial do Projeto — Sensoriamento de Poses Humanas com WiFi

> ### 📌 Documento histórico — primeira fase do projeto
>
> Esta análise é sobre o dataset **Wi-Pose**, que foi **abandonado em 10/09/2026**
> justamente por causa das limitações descritas aqui: anotação 2D e ausência da
> fase do sinal. O projeto migrou para o **MM-Fi**.
>
> **Para o estado atual, veja [`analise-mmfi.md`](analise-mmfi.md).**
>
> Mantido no repositório porque documenta *por que* a troca foi feita — o que é
> material da tarefa (a), revisão bibliográfica.


**Disciplina:** Ações Integradas de Ciência e Tecnologia I (AICT I)
**Código da tarefa:** `RB-01` (revisão bibliográfica e enquadramento do projeto)
**Data:** 03/09/2026
**Equipe:** 3 integrantes

> Documento de enquadramento. Números de terceiros vêm dos artigos e do README do dataset; onde a evidência é indireta, isso está dito na linha.
>
> **Atualizado em 03/09/2026, após o dataset ser baixado e inspecionado.** As perguntas que estavam em aberto na §4.2 foram respondidas rodando `scripts/inspecionar.py` sobre os 166.600 arquivos reais — o resumo do que mudou está na **§0**, e as seções técnicas foram atualizadas de acordo.

---

## 0. O que a inspeção do dataset confirmou (03/09/2026)

O dataset foi baixado, extraído e inspecionado. Esta seção resume o que mudou; o
detalhe está nas seções indicadas.

| Pergunta que estava em aberto | Resposta | Efeito |
| --- | --- | --- |
| Formato dos arquivos | ✅ MATLAB **v7.3 / HDF5** | `scipy.io.loadmat` não abre; usar `h5py` (§4.1) |
| Nomes das variáveis | ✅ **`CSI`** e **`SkeletonPoints`** — os do README (`csi_serial`, `jointsVector`) **não existem**, e `jointsMatrix` **não é distribuído** | A inferência de segunda mão estava certa (§4.1) |
| O CSI é complexo? | ✅ **Não — é amplitude real** (`float64`, zero negativos em 300 arquivos) | **Não há fase.** Rescopa as tarefas (b) e (d) (§5.2) |
| Forma do CSI | ✅ `(3, 3, 30, 5)` lido com `h5py` | Confirma a ordem de eixos prevista (§4.1) |
| Ordenação de `SkeletonPoints` | ✅ **Blocos** `[18 y][18 x][18 c]` — **y vem antes de x** — a confiança fica em [0, 1]; a hipótese intercalada dava "confiança" de 434 | Confirma `DT-Pose`/`HPE-Li`; `CSI-VQ` está errado (§4.1) |
| O `id` é voluntário ou sequência? | ✅ **Sequência** — exatamente 120 ids distintos por ação, de 1 a 120 | **Split por voluntário é impossível.** Split por sequência é o cenário base (§5.1) |
| O split oficial vaza? | ✅ **Não.** 1.152 sequências em `Train`, 288 em `Test`, **zero em comum** | Melhor que o previsto — o split oficial respeita agrupamento (§5.1) |
| Classe majoritária | ✅ **8,55%** (`walk`), contra 8,33% de acaso | Dataset quase balanceado (§7) |

**Dois achados novos, que a análise anterior não previa:**

1. 🆕 **10,6% dos pacotes são cópias exatas do pacote anterior** (IC 95% ±0,4 pp,
   sobre 180 sequências sorteadas com semente 42 — cerca de 17,6 mil pacotes). O
   dataset estica cada sequência repetindo o último quadro até um comprimento
   alvo, quase sempre a partir do quadro ~105. Num split aleatório por pacote,
   **cópias idênticas cairiam em treino e teste ao mesmo tempo** — é a prova
   concreta do vazamento descrito na §5.1. Detectáveis com
   `src.dados.indices_duplicados`.

   *(Cuidado ao medir: amostrar só as sequências `_001` dá ~16%, porque elas são
   mais longas e mais preenchidas que a média. O script usa amostra aleatória.)*

2. 🆕 **17,6% das juntas anotadas têm confiança abaixo de 0,3** (mediana geral:
   0,632). O "ground truth" é ruidoso numa fração relevante dos quadros, o que
   reforça a necessidade do critério de descarte/ponderação da §5.5.

**As sequências variam muito mais do que o estimado:** de **50 a 216 quadros**
(2,5 a 10,8 s), mediana 110 e média 115,7 — não a faixa estreita de ~116 que a
média sugeria. A regra de janelamento da §5.6 precisa cobrir esse intervalo
inteiro: uma janela de 1 s (20 quadros) rende 2 janelas na sequência mais curta e
10 na mais longa.

---

## 1. O que a disciplina cobra

O `tema.pdf` define o foco como *"processamento digital de sinais de rádio (RF) e modelagem de modelos preditivos para estimação de movimentos corporais sem o uso de câmeras"*, e lista cinco tarefas sugeridas:

| Código | Tarefa | Natureza |
| --- | --- | --- |
| `RB` | a) Revisão bibliográfica — Wi-Pose (atividades predefinidas) × Winect (free-form) | Escrita |
| `ED` | b) Manipulação das estruturas de dados — carregar CSI, plotar amplitude e fase | Código / DSP |
| `CM` | c) Classificação de movimentos — modelo de ML para discriminar as ações | Código / ML |
| `PC` | d) Pré-processamento e calibração — filtragem espacial/temporal, remoção de offset de fase | Código / DSP |
| `AC` | e) Análise crítica — matrizes de confusão, impacto de subportadoras OFDM ruidosas | Análise |

Repare na estrutura: **(b) → (d) → (c) → (e)** é uma cadeia de dependência, e **(a)** corre em paralelo. A tarefa (e) não é um adendo — ela é o fecho analítico que dá sentido ao resto, e é a única que exige explicitamente um resultado *negativo bem medido* (quais subportadoras atrapalham).

O verbo central da tarefa (c) é **classificar** — não reconstruir esqueleto. Isso importa muito, e a seção 3 explica por quê.

⚠️ **Requisito fácil de perder:** a Descrição Geral do `tema.pdf` pede *"identificar **e visualizar** os movimentos corporais"*. Classificação sozinha não visualiza nada. Ver o item de visualização na Rota A (§7).

---

## 2. Os dois artigos de referência

### 2.1 Wi-Pose (Jiang et al., MobiCom 2020) — "Towards 3D Human Pose Construction Using WiFi"

**Problema:** reconstruir esqueletos 3D (17 juntas, tronco + membros) a partir de WiFi comercial, sem câmera e sem sensor vestível.

**Ideias centrais:**

1. **Cinemática direta (*forward kinematics*) como camada da rede.** Em vez de estimar cada junta de forma independente — o que produz esqueletos com braços de comprimentos absurdos —, o modelo aprende a **rotação** de cada segmento em relação ao pai na árvore do esqueleto, representada por **quatérnions unitários**, e aplica essas rotações a uma estrutura óssea dada. O comprimento do osso fica fixo por construção, então a pose sai anatomicamente plausível de graça.
2. **Arquitetura:** 4 camadas de CNN (batch norm + Leaky ReLU α=0,02 + dropout 0,2; filtros 64/128/64/1) → 3 camadas de LSTM (estado oculto 544, dropout 0,1) → camada de cinemática direta. Otimizador Adam.
3. **Três perdas somadas** (`J = Lp + β·Ls + γ·Lr`, com β=γ=1,0):
   - `Lp` — posição (L2 entre junta prevista e ground truth);
   - `Ls` — suavidade (norma de Huber sobre a diferença entre poses consecutivas), para o esqueleto não tremer;
   - `Lr` — rotação (erro na posição *relativa* ao pai), para o erro não acumular ao descer a árvore.
4. **Generalização entre ambientes via 3D Velocity Profile (3DVP).** Extensão 3D do *Body-coordinate Velocity Profile*. O CSI cru carrega assinatura do ambiente; o perfil de velocidades não, porque objetos estáticos têm velocidade zero. Pipeline: multiplicação conjugada do CSI (cancela o offset de fase de CFO/SFO/atraso de detecção) → subtração da média (remove o componente estático dominante) → STFT → perfil de *Doppler Frequency Shift* por link → *compressed sensing* para resolver `D⁽ᵐ⁾ = c⁽ᵐ⁾A⁽ᵐ⁾V` e obter o tensor de velocidades 9×9×9.

**Testbed:** 1 transmissor + **9 antenas receptoras distribuídas em 3 hospedeiros** (Intel 5300, Linux 802.11 CSI Tool), canal 165 (5,825 GHz), 20 MHz, **1000 pacotes/s**. Ground truth: **VICON** com 21 câmeras infravermelho e 17 marcadores refletivos, erro < 2 mm, 10 Hz. Sincronização por NTP (erro médio 7 ms).

**Dados:** 10 voluntários, **16 atividades** — levantar mão esquerda/direita a 45/90/180°, levantar as duas mãos a 90/180°, varrer mão esquerda/direita a 45/90°, varrer as duas mãos a 90°, levantar perna, acenar, andar no lugar. 1 minuto por atividade por sujeito.

**Resultados** (erro médio de localização por junta, cenário básico, média sobre 17 juntas):

| Modelo | Entrada | Erro global |
| --- | --- | --- |
| RFPose3D (baseline) | CSI | 43,6 mm |
| **WiPose** | **CSI** | **28,3 mm** |
| RFPose3D | 3DVP | 76,7 mm |
| WiPose | 3DVP | 36,7 mm |

Juntas de braço (cotovelo e mão, índices 3/4/6/7) erram muito mais que o resto — 59, 105, 52 e 90 mm — por área de reflexão pequena e amplitude de movimento grande. A junta raiz erra **0 mm por construção** (é o ponto fixo da cinemática direta) e as juntas de tronco ficam em 7–20 mm; isso puxa a média global para baixo, o que importa na comparação da §2.3. O 3DVP piora no domínio de origem (perde informação postural) mas **degrada muito menos ao trocar de ambiente**, que é justamente o ponto dele.

**Detalhe que muda o nosso planejamento:** para produzir o resultado de manchete (28,3 mm com entrada de CSI), o artigo **descarta a fase por completo** — §6.1.1: *"we remove the phase offset through calculating the absolute value of the raw CSI data"*. A multiplicação conjugada da §4.1 existe apenas no caminho do 3DVP. Ou seja: o melhor número do Wi-Pose vem de **amplitude pura**. Ver §5.2.

**Limitação declarada:** o modelo é treinado sobre um **dicionário fechado de atividades**. É essa limitação que o Winect ataca.

### 2.2 Winect (Ren et al., IMWUT 2021) — "3D Human Pose Tracking for Free-form Activity"

**Problema:** o mesmo rastreamento 3D, mas para movimento **livre**, sem dicionário pré-treinado.

**Abordagem — sem deep learning como espinha dorsal, e sim processamento de sinal clássico:**

1. **Identificação de membros em movimento** por **AoA 2D** (azimute + elevação) com **array em L** e algoritmo **MUSIC**, melhorado por diversidade espacial (3 antenas TX) e de frequência (subportadoras OFDM) para compensar as poucas antenas do Intel 5300. Subtração do ambiente estático inspirada em subtração espectral.
2. **Separação de sinais dos membros** como problema de **Blind Source Separation**, resolvido por **ICA**. Os autores validam empiricamente as premissas do ICA: correlação entre movimentos de membros diferentes < 0,08 (independência) e distribuições não-gaussianas.
3. **Seleção de subportadora** pelo *movement energy ratio*: FFT da amplitude do CSI, razão entre a energia nas frequências de movimento (0–20 Hz) e a energia total. Escolhe-se a subportadora com maior razão. Ideia que consideramos e **descartamos** como critério para a tarefa (e) — ela não sobrevive à taxa de 100 Hz deste dataset; ver §7.
4. **Decomposição em juntas** por modelo cinemático de membro, treinado (ResNet-18) sobre **dois** conjuntos de pontos pareados: a nuvem de **membro** derivada das medidas de CSI e a nuvem de **juntas** registrada em paralelo pelo Kinect 2.0. O artigo diz ">24.000 pontos" condicionado a cada voluntário executar ≥20 min de atividade — não fica claro se é por voluntário ou no total.

**Testbed:** 5 laptops — 1 TX e **4 RX, totalizando 12 antenas receptoras**, uma delas montada como array em L calibrado a λ/2. **5,32 GHz, 40 MHz de banda, 1000 pacotes/s.** Kinect 2.0 como ground truth a 10 Hz, sala e quarto de uma casa. 6 voluntários (3M/3F), segmentos de 8 s, >900 atividades.

**Resultados** (erro de localização, média sobre **8 juntas de membro**: cotovelos, punhos, joelhos, tornozelos):

| Sistema | Erro global |
| --- | --- |
| **Winect** | **4,6 cm** |
| WiPose (reproduzido, geral) | 10,1 cm |
| WiPose — só atividades predefinidas | 4,3 cm |
| WiPose — só atividades free-form | 16,0 cm |

A leitura correta desta tabela é a tese do artigo: **em atividades predefinidas o WiPose empata com o Winect (4,3 vs 4,6 cm); em movimento livre o WiPose desmorona (16,0 cm)**. O Winect paga esse ganho com hardware mais exigente (array em L calibrado, 4 receptores) e um pipeline de DSP bem mais pesado.

### 2.3 O contraste que a tarefa (a) pede

| Eixo | Wi-Pose (MobiCom'20) | Winect (IMWUT'21) |
| --- | --- | --- |
| Paradigma | Aprendizado ponta a ponta | Modelo físico + DSP + ML localizado |
| Espaço de atividades | Dicionário fechado | Livre |
| Conhecimento prévio injetado | Esqueleto (cinemática direta / quatérnions) | Cinemática de membros + física da propagação |
| Ground truth | VICON (2 mm) | Kinect 2.0 |
| Generalização | Via representação (3DVP) | Via modelo físico (independente de ambiente por construção) |
| Hardware receptor | 9 antenas RX em 3 hospedeiros; 5,825 GHz, **20 MHz** | 12 antenas RX em 4 hospedeiros, uma em array em L calibrado; 5,32 GHz, **40 MHz** |
| Erro reportado ⚠️ | 2,83 cm — média sobre **17** juntas, incluindo raiz fixa (0 mm) e tronco | 4,6 cm — média sobre **8** juntas de **membro** |
| Fraqueza | Não vê o que não treinou | Depende de estimar corretamente o nº de membros em movimento; ICA degrada com muitos membros (6,7 cm com 3 membros) |

⚠️ **"2,83 cm vs 4,6 cm" não é uma comparação válida** — são conjuntos de juntas diferentes (17, incluindo a raiz de erro zero, contra 8 juntas de membro), testbeds diferentes e ground truths de qualidade muito diferente (VICON <2 mm vs Kinect 2.0). Citar esse par como se o WiPose fosse mais preciso é o erro que quem escrever a §2 do relatório precisa evitar.

✅ **A comparação válida está na Tabela 2 do Winect**, onde os autores reimplementaram o WiPose e mediram os dois no mesmo testbed, nas mesmas 8 juntas de membro. Vale a pena olhar as 8, porque a média esconde a estrutura:

| Junta | WiPose-predefinido | Winect |
| --- | --- | --- |
| Cotovelo E | 4,9 | **4,2** |
| Punho E | 6,1 | **5,1** |
| Cotovelo D | 5,2 | **4,8** |
| Punho D | 5,8 | **4,9** |
| Joelho E | **3,5** | 4,1 |
| Tornozelo E | **4,0** | 4,4 |
| Joelho D | **2,1** | 4,3 |
| Tornozelo D | **2,5** | 4,7 |
| **Média** | **4,3** | **4,6** |

Não é empate difuso: é uma **divisão limpa**. O Winect vence nos **4 de 4** pontos de braço; o WiPose vence nas 4 de perna, por até 2× (2,1 vs 4,3 no joelho direito). A vantagem global do WiPose-predefinido vem **inteiramente das pernas**. E 0,3 cm de diferença na média, sob ground truth de Kinect 2.0 (cujo erro articular próprio é da ordem de centímetros), não sustenta ordenação nenhuma — as CDFs das Figuras 17/18 do artigo confirmam: mediana de ~3,9 cm e percentil 80 de ~7,0 cm **para ambos**.

**A diferença que sustenta** aparece em movimento livre: WiPose vai a **16,0 cm**, Winect fica em **4,6 cm**. Essa — e só essa — é a tese defensável do contraste.

*(Notas de rigor para P3: a reimplementação do WiPose foi feita pelo grupo concorrente, no testbed dele; e o ground truth Kinect engole diferenças da ordem de 0,3 cm. A diferença de largura de banda entre os dois papers **não** é confundidor aqui — nesta tabela ambos rodaram no testbed do Winect, a 40 MHz.)*

---

## 3. 🚨 Achado crítico: o dataset **não é** o dataset do artigo Wi-Pose

Este é o ponto mais importante deste documento, e ele muda o planejamento.

Existem duas coisas diferentes chamadas "Wi-Pose": o **artigo** do MobiCom 2020 (SUNY Buffalo) e o **dataset** [`NjtechCVLab/Wi-PoseDataset`](https://github.com/NjtechCVLab/Wi-PoseDataset) (Nanjing Tech University), que é o que vamos usar.

| | Artigo Wi-Pose (MobiCom 2020) | Dataset `NjtechCVLab/Wi-PoseDataset` |
| --- | --- | --- |
| Grupo | SUNY Buffalo (Jiang, Xue, Su et al.) | Nanjing Tech University (NjtechCVLab) |
| Ground truth | **VICON**, 17 marcadores, **3D**, erro < 2 mm | **AlphaPose sobre vídeo**, 18 pontos, **2D + confiança** |
| Voluntários | 10 | 12 |
| Ações | 16 (gestos de braço/perna granulares) | 12 (bend, circle, crouch, jump, pull, push, run, sit down, stand up, throw, walk, wave) |
| Antenas | 1 TX + 9 RX **distribuídas** | 3 TX × 3 RX (dois dispositivos comuns) |
| Taxa de CSI | ~1000 Hz | **100 Hz** (vídeo a 20 Hz; 5 CSI por quadro) |
| Publicação associada | MobiCom 2020 | **CSI-Former**, Entropy (MDPI) 2023 |

O CSI-Former declara publicar o dataset (*"we publish the new dataset Wi-Pose at …"*, *"to the best of our knowledge, there is no such public dataset yet"*) e **não cita o artigo do MobiCom em nenhuma das 61 referências**. Do outro lado, o artigo do MobiCom não traz declaração de release nem URL de repositório — não localizamos os dados dele publicamente (afirmação negativa, não testada com os autores). Seja como for, a Rota C (§7) está fechada por outras três razões técnicas verificadas.

**Quem confunde o quê, para sermos justos:**

- O `tema.pdf` lista as duas coisas **separadamente** e rotula corretamente o MobiCom como *"Artigo **Conceitual** Wi-Pose"*. Descreve o dataset com precisão (AlphaPose, 18 pontos, 12 voluntários, 12 ações). O único deslize é a palavra **"3D"** em *"anotações de pose real em 3D obtidas pelo … Alphapose"* — AlphaPose é estimador **2D**.
- O **nosso** `README.md` é que funde as duas fontes, ao prometer "estimação de movimentos e poses corporais **3D**". O `requirements.txt` repete no comentário do `plotly` ("visualização 3D do esqueleto"). Isso precisa ser corrigido — é o passo 0 da §9.
- Cuidado extra ao ler a literatura: vários repositórios de terceiros chamam o dataset da Nanjing Tech de "WiPose Dataset". Qualquer benchmark que cite esse nome tem que ser lido com atenção a qual dos dois se refere.

**Consequências práticas:**

1. **Não temos ground truth 3D.** Qualquer promessa de "reconstrução 3D" no nosso relatório seria falsa a partir deste dado. O que dá para fazer é **pose 2D** (regressão) ou **classificação de ação**.
2. **Não dá para reproduzir o pipeline do MobiCom com este dataset.** O 3DVP exige (i) a **geometria conhecida** das antenas — a Eq. 11 precisa das posições de TX e de cada RX, que não são publicadas — e (ii) uma **janela temporal** suficiente para a STFT; aqui são 5 amostras por anotação, ou **50 ms**.
3. **O baseline realista é o CSI-Former, não o WiPose.** O artigo do dataset reporta PCK@5 = 0,5505 (contra 0,5231 do ResNet) e PCK@50 médio 0,8417 — em **regressão de pose 2D**, não em classificação. Para a tarefa (c), que é classificação, **não localizamos baseline publicado** sobre este dataset — vamos construir o nosso. (Ressalva: a busca foi rasa. Há pipeline público que já trata a ação como rótulo de primeira classe — o `DT-Pose` mantém um `action_dict` com as 12 classes —, então vale uma busca mais séria antes de afirmar isso no relatório.)

**Nada disso invalida o projeto.** Os dois artigos continuam sendo a base conceitual correta — a revisão bibliográfica (a) é sobre eles. O que muda é que o **trabalho experimental** roda sobre um dataset diferente, com anotação 2D e ações de corpo inteiro, e isso precisa estar escrito no relatório. Essa discrepância é, ela mesma, material de análise crítica de primeira qualidade para a tarefa (e).

---

## 4. O que o dado realmente entrega

### 4.1 ⚠️ Divergência entre o README e o arquivo distribuído

O README do repositório declara três variáveis:

| Variável (README) | Forma | Conteúdo |
| --- | --- | --- |
| `csi_serial` | `5 × 30 × 3 × 3` | 5 amostras temporais de CSI (janela de 50 ms), 30 subportadoras OFDM, 3 TX × 3 RX |
| `jointsVector` | `18 × 4` | 18 juntas `(x, y, c, c)` |
| `jointsMatrix` | — | Deslocamentos relativos entre juntas |

**Porém**, três carregadores públicos independentes que consomem este dataset (DT-Pose, HPE-Li/ECCV2024, CSI-VQ-Compression) concordam que o arquivo realmente distribuído (`Wi-Pose.rar` → pastas `Train/` e `Test/`) contém `.mat` no formato **v7.3 / HDF5** com **duas** variáveis:

| Variável (arquivo real, segundo terceiros) | Forma | Conteúdo |
| --- | --- | --- |
| `CSI` | `5 × 30 × 3 × 3` | mesma forma; **natureza (complexo ou amplitude real) não confirmada** — ver abaixo |
| `SkeletonPoints` | vetor de **54** | 18 juntas × 3 valores `(x, y, c)` — **a ordenação está em disputa**, ver abaixo |

Quanto a **`jointsMatrix`**: o README oficial afirma que ela está no pacote, e nenhum dos três carregadores a lê. Mas não usar não é evidência de não existir — nenhum dos três precisa dela. **Fica em aberto**, e `print(f.keys())` no passo 2 responde.

> ✅ **CONFIRMADO no dado.** O que era inferência de segunda mão foi verificado abrindo os arquivos: as variáveis são mesmo `CSI` e `SkeletonPoints`, o formato é v7.3/HDF5, e `jointsMatrix` **não existe** no pacote distribuído. O loader (`src/dados.py`) trata tudo isso.

**Ordem dos eixos do CSI em disco.** MATLAB é *column-major* e HDF5 inverte: lendo `CSI` com `h5py` cru, o array chega como **`(3, 3, 30, 5)`**; `mat73` **restaura** a ordem MATLAB `(5, 30, 3, 3)`. **O contrato do loader (§8) tem que nomear o leitor** — o mesmo `reshape` está certo num e errado no outro, silenciosamente.

Quanto ao alvo, os carregadores públicos **não convergem**: `HPE-Li` e `CSI-VQ` usam `(9, 30, 5)`, mas o `DT-Pose` faz `.transpose(3,2,1,0).reshape(3, 90, 5)` — que **funde antena com subportadora**, exatamente o que a §5.4 diz destruir a estrutura espectral. **Adotamos `(9, 30, 5)`** — **pares TX×RX**, subportadoras, tempo — não por consenso, mas por mérito: mantém antena e subportadora em eixos separados, pré-requisito para a tarefa (e) (não dá para analisar impacto de subportadora num eixo fundido com antena).

⚠️ **Mas o eixo 9 funde TX com RX, e isso tem consequência.** São 3 TX × 3 RX. Duas prescrições deste documento — a **multiplicação conjugada entre antenas do mesmo receptor** (§5.2) e o **produto conjugado entre antenas RX do mesmo rádio** (§5.8) — exigem saber **qual subconjunto dos 9 compartilha rádio**. Como `h5py` e `mat73` entregam ordens diferentes, a posição de TX vs RX no índice achatado é indeterminada até alguém verificar. É o mesmo modo de falha silenciosa da §5.4, agora dentro do próprio contrato que deveria protegê-lo. **O contrato (§8) tem que fixar a convenção** — por exemplo `tx = idx // 3`, `rx = idx % 3` — e o passo 2 tem que confirmá-la no arquivo real.

**Duas questões que podiam quebrar o pipeline em silêncio — ambas já resolvidas:**

1. ✅ **RESOLVIDO: o `CSI` é amplitude real, não complexo.** `float64`, sem dtype composto, zero valores negativos em 300 arquivos. **O dataset não distribui a fase.** Isso rescopa as tarefas (b) e (d): não há "amplitude e fase" para plotar nem "offset de fase" para remover. Ver §5.2 e o passo 4 da §9.
2. ✅ **RESOLVIDO: `SkeletonPoints` está em blocos, e `y` vem antes de `x`.** A dúvida era entre leitura intercalada `(x,y,c)` por junta (usada pelo `CSI-VQ`) e leitura em blocos (usada por `DT-Pose` e `HPE-Li`). Duas verificações decidiram: **(a)** o terceiro bloco cai em [0, 1] — é a confiança; a hipótese intercalada daria "confiança" de 434; **(b)** teste anatômico sobre 1.020 pacotes — a distância pescoço→quadril (vertical numa pessoa em pé) é maior no bloco 0 em **99,4%** dos casos, e a distância ombro→ombro (horizontal) é maior no bloco 0 em apenas **6,4%**.

   Ou seja, o layout real é **`[18 y][18 x][18 confiança]`** — `y` **antes** de `x`, ao contrário do que o nome "(x, y, c)" do README sugere. `src/dados.py` reordena e devolve `(x, y, confiança)`.

**Volumetria:** 166.600 pacotes. O README diz "~13.200 por voluntário", mas 12 × 13.200 = 158.400 ≠ 166.600 — o README é auto-inconsistente. Como 132.847 + 33.753 = 166.600 fecha exatamente, o número confiável é **166.600 no total, ~13.900 por voluntário**.

**Dimensionalidade por amostra:** 5 × 30 × 3 × 3 = 1.350 valores. O tamanho em disco depende de duas incógnitas — complexo ou real, `float64` (padrão do MATLAB) ou `float32`:

| | `float32` | `float64` |
| --- | --- | --- |
| Real (amplitude) | ~0,9 GB | ~1,8 GB |
| Complexo | ~1,8 GB | ~3,6 GB |

Cabe em memória em qualquer um dos casos, mas a diferença é 4× — o passo 2 fecha isso.

**Split "oficial":** o pacote já vem separado em `Train/` (132.847) e `Test/` (33.753) — 79,74%, coerente com um 80/20. O artigo não declara o critério, mas ✅ **a medição mostra que ele é bem-feito**: 1.152 sequências em `Train`, 288 em `Test`, **interseção zero**. E é sistemático — `Test` leva os ids ≡ 9 e 10 (mod 10) de cada ação. Dá para usá-lo como ponto de partida sem medo de vazamento de quadro. Ver §5.1.

### 4.2 Perguntas em aberto que só o download responde

Estas são o primeiro item do plano de execução, em ordem de impacto:

1. ✅ **O `CSI` é amplitude real** (`float64`, nunca negativo). A tarefa (d) não tem fase para calibrar.
2. ✅ **As variáveis são `CSI` e `SkeletonPoints`**; `jointsMatrix` não existe no pacote distribuído.
3. ⚠️ **O `id` do nome do arquivo quase certamente é a sequência, não o voluntário.** A convenção reportada é `<ação>_<id de 3 dígitos>-frame<3 dígitos>.mat`, e o exemplo literal encontrado no `DT-Pose` é **`bend_029-frame043.mat`** — **029 > 12**, logo o campo não pode ser um índice de voluntário num conjunto de 12. A leitura provável é sequência global 1–120 por ação (12 voluntários × 10 repetições). *(O `DT-Pose` chama a variável de `person` no código, mas isso é nome de variável, não evidência contra o 029.)* **Confirmação em uma linha depois do download:** contar valores distintos do campo dentro de uma ação — ~120 ⇒ sequência, ~12 ⇒ voluntário. **Antes de desistir do split por sujeito, teste o agrupamento:** se o `id` é sequência global 1–120 estruturada como 12 voluntários × 10 repetições, então `⌈id/10⌉` é candidato direto a índice de voluntário — verificável checando se as estatísticas de CSI se agrupam em blocos de 10 ids consecutivos. **Só se nem isso funcionar** o split por voluntário fica irrecuperável, e o número passa a ser obrigatoriamente intra-sujeito (§5.1).
4. ✅ **Blocos, com `y` antes de `x`**: `[18 y][18 x][18 confiança]` (§4.1).
5. ✅ **`h5py`** — decidido no contrato. `mat73` foi dispensado justamente por restaurar a ordem de eixos do MATLAB e reabrir a divergência.
6. ❓ **`SkeletonPoints` está em pixels de qual resolução?** *Ainda em aberto.* As coordenadas observadas vão de ~16 a ~616, compatíveis com um quadro retrato de ~480×640, mas a resolução não é declarada. Dois carregadores de terceiros multiplicam por 0,001 antes de usar. **Só importa para a Rota B** (regressão de pose, onde o PCK é normalizado); a classificação não depende disso.

---

## 5. Riscos metodológicos — leia antes de escrever qualquer modelo

### 5.1 Vazamento entre treino e teste (o risco número um)

Pacotes `.mat` consecutivos vêm de quadros de vídeo a 20 Hz da **mesma repetição**, executada pela **mesma pessoa**. O CSI-Former declara 5–6 s por repetição, mas a medição no dado desmente a faixa estreita: as 1.440 sequências vão de **50 a 216 quadros** (2,5 a 10,8 s), com mediana 110 e média 115,7 (§0). Dividir treino/teste aleatoriamente por pacote coloca quadros vizinhos — 50 ms de diferença, mesma pessoa, mesma roupa, mesma posição na sala — nos dois lados da divisão. O modelo memoriza a assinatura estática do canal e a acurácia infla sem medir generalização nenhuma.

*(As "10 repetições por ação por voluntário", que com 12 voluntários e 12 ações dão 1.440 sequências, vêm do README do GitHub.)*

Em datasets de CSI isso é o modo de falha clássico. **Não use um limiar de acurácia como diagnóstico** — não localizamos baseline publicado de classificação neste dataset (§3, com a ressalva de que a busca foi rasa) para calibrar o que é "alto demais". O diagnóstico é a **diferença entre os splits**: se o split aleatório por pacote der muito acima do split por sequência, a distância entre os dois *é* a medida do vazamento. Rode o split por sequência e o aleatório por pacote e compare os dois.

**Regra da equipe, em ordem de preferência:**

1. **Split por sequência/repetição** — as 120 sequências de cada ação vão inteiras para um lado só. **Este é o cenário base**, porque o `id` do nome do arquivo quase certamente indexa sequência e não voluntário (§4.2, pergunta 3: o exemplo `bend_029` já tem 029 > 12).
2. **Split por voluntário** (pessoas disjuntas) — seria o padrão-ouro, por medir generalização para pessoa nova. **Provavelmente indisponível diretamente** — o `id` parece indexar sequência. Mas antes de abandonar o padrão-ouro, teste o agrupamento em blocos de 10 (§4.2, pergunta 3); se ele recuperar o sujeito, promova este split a principal.
3. **Split aleatório por pacote** — **nunca** como resultado principal.

Operando no cenário base (caso 1), o relatório precisa dizer com todas as letras que o número é **intra-sujeito** e **não** mede generalização para pessoa nova — as mesmas 12 pessoas aparecem no treino e no teste, em repetições diferentes. Essa é uma limitação do dataset, não do trabalho, e declará-la é parte da tarefa (e). E vale reportar os splits disponíveis lado a lado: **a diferença entre eles é, sozinha, um dos melhores resultados da tarefa (e).**

### 5.1.1 ⚠️ O tamanho amostral efetivo é 1.440, não 166.600

Esta é a consequência da regra de split que é mais fácil de esquecer, e ela reduz o poder estatístico de tudo o que vem depois.

Se a unidade independente é a **sequência**, então o dataset tem **12 ações × 120 sequências = 1.440 unidades**, não 166.600. Um teste de 20% é ~288 sequências, **~24 por classe**. Com a agregação de 1 s (§5.6), são ~5 a 6 janelas por sequência — **~8.300 janelas**, ainda amarradas em 1.440 blocos correlacionados.

Três consequências práticas:

- **Todo número reportado precisa de intervalo de confiança**, calculado por *bootstrap* **sobre sequências** (reamostrar sequências inteiras, nunca pacotes).
- **As duas barras de erro são muito diferentes, e confundi-las é fácil.** Para a **acurácia global / F1-macro**, sobre as 288 sequências de teste, o IC fica na ordem de **±6 pp**. Para o **recall de cada classe**, calculado sobre ~24 sequências, o IC vai para **±18 a 20 pp**. Isso importa porque recall por classe e matriz de confusão são o entregável de manchete da tarefa (e): com ±20 pp, **quase nenhuma célula fora da diagonal é distinguível de outra**. A matriz 12×12 sobre 288 sequências sustenta afirmações sobre **blocos** ("as classes de perna se confundem entre si"), não sobre pares individuais. Para investigar um par específico — como o da §5.8 — use uma comparação binária dirigida, não a célula da matriz.
- **Para a ablação de subportadoras, use bootstrap *pareado*.** A ablação compara modelos sobre **as mesmas** sequências de teste, então a quantidade que precisa de IC é a **diferença** (Δacurácia entre com e sem as k subportadoras), não cada ponto isolado. No bootstrap pareado — reamostrar sequências e avaliar as duas configurações na mesma reamostragem — a variância compartilhada cancela, e o IC da diferença fica tipicamente 3–5× mais estreito que os ±6 pp. Sem isso, a curva de 30 pontos vira 2 ou 3 pontos utilizáveis à toa. Plote a banda de confiança junto da curva. **Contingência:** se nem a diferença pareada separar, o resultado da tarefa (e) passa a ser *"a seleção de subportadoras não tem efeito mensurável neste N"* — achado legítimo e reportável.

**De onde sai a validação.** O pacote traz só `Train/` e `Test/` — não há conjunto de validação. Ele é recortado **de dentro de `Train/`, pela mesma regra de split** adotada (por sujeito ou por sequência). Recortar a validação aleatoriamente por pacote reintroduz exatamente o vazamento descrito acima, só que pela porta dos fundos, e faz toda escolha de filtro e hiperparâmetro ser feita sobre um número inflado.

**Disciplina de decisão:** **nenhuma decisão** — filtro, seleção de subportadoras, hiperparâmetro, arquitetura — é tomada olhando o teste. Decidir é na validação. Isso não proíbe *reportar* vários números de teste no fim (os três splits, ou o split oficial ao lado do nosso); proíbe **escolher** com base neles. Ablação de subportadoras medida no teste e depois reportada naquele mesmo teste é *test-set fishing* e não se defende numa banca.

### 5.2 A fase crua do CSI é inutilizável — **se houver fase**

> ⚠️ Esta seção inteira depende da pergunta 1 da §4.2. Se o `CSI` distribuído já for amplitude real, não há fase para calibrar e a tarefa (d) se rescopa para filtragem temporal e seleção espacial/espectral. Verifique antes de escrever uma linha de `src/dsp.py`.

A fase medida vem contaminada por *Carrier Frequency Offset*, *Sampling Frequency Offset* e atraso de detecção de pacote — o termo `e^{jε(f,t)}` da Eq. 9 do MobiCom — e ainda envolvida em ±π. Plotar fase crua e concluir "a fase não tem informação" é errar o diagnóstico. Antes de qualquer conclusão sobre fase:

- `unwrap` ao longo das subportadoras;
- **sanitização linear** (regressão linear sobre as 30 subportadoras, subtrair a reta) — remove o termo comum de offset;
- ou **multiplicação conjugada entre antenas do mesmo receptor** — cancela os offsets comuns por serem idênticos nas antenas de um mesmo rádio, e **preserva o sinal do Doppler relativo**, que a amplitude descarta.

Isto é literalmente a tarefa (d), e tem resposta conhecida na literatura — bom lugar para começar.

**Expectativa calibrada:** o resultado de manchete do Wi-Pose (28,3 mm) usa **amplitude pura, fase descartada** (§2.1). Ou seja, é perfeitamente possível gastar o semestre em sanitização de fase e não bater a amplitude. Por isso o baseline da tarefa (d) é **amplitude sem fase** — é ele que a fase precisa superar para justificar o custo. Se não superar, isso é um resultado legítimo e reportável.

### 5.3 Amplitude precisa de filtragem, mas com parcimônia

Outliers impulsivos pedem **filtro de Hampel**; ruído de alta frequência pede **passa-baixa Butterworth** ou *denoising* por wavelet (`PyWavelets` já está no `requirements.txt`). **Contraponto — mas leia com desconfiança:** os autores do CSI-Former relatam que Butterworth e mediana de 10 pontos **degradaram** as características de pose neste mesmo dataset. ⚠️ Só que o artigo declara ter usado um Butterworth com *"sampling rate of 1000 samples per second"* sobre um CSI publicado a **100 Hz**. Ou é erro de parametrização (frequência de corte normalizada 10× errada), ou o filtro foi aplicado **antes** de uma subamostragem de 1000 para 100 Hz — e aí está correto. Não conseguimos determinar qual. **Confirme no artigo antes de levar a crítica ao relatório**, e até lá **trate o resultado deles como não conclusivo** — e registre isso no relatório, porque é boa análise crítica para a tarefa (e). O plano não muda: medir o efeito de cada filtro contra o baseline sem filtro, que é exatamente o que a tarefa (d) pede.

### 5.4 Confusão de eixos no tensor

`5 × 30 × 3 × 3` tem duas dimensões de tamanho 3 lado a lado (TX e RX). Um `reshape` sem `transpose` mistura subportadora com antena e destrói a estrutura espectral **sem levantar exceção nenhuma**. E o risco é concreto, não hipotético: `h5py` e `mat73` entregam o mesmo arquivo em ordens diferentes (§4.1), então o mesmo `reshape(9,30,5)` está certo num e errado no outro.

Convenção da equipe: o loader fixa o leitor e devolve sempre `(9, 30, 5)`; toda função que transforma o tensor documenta forma de entrada e de saída na docstring; e o teste de fumaça verifica a forma contra uma fixture conhecida.

### 5.5 Juntas com baixa confiança

A terceira coluna de `SkeletonPoints` é a confiança do AlphaPose. Junta com `c` baixo é **ruído de anotação**, não verdade de campo — o "ground truth" aqui já é a saída de outro modelo, com erro próprio. Se formos para regressão de pose, é obrigatório definir e declarar um critério (descartar abaixo de um limiar, ou ponderar a perda por `c`). Para classificação de ação isso importa menos, mas vale como filtro de qualidade de amostra.

### 5.6 Janela temporal de 50 ms

Distinguir `walk` de `run` a partir de 50 ms é difícil por natureza — a diferença entre elas é de **cadência**, que não cabe nessa janela. Duas saídas: (i) agregar N pacotes consecutivos da mesma repetição numa janela maior (20 pacotes ≈ 1 s) antes de classificar, ou (ii) classificar por pacote e reportar também a acurácia agregada por votação sobre a repetição. **A opção (i) é mais forte, mas cria dependência entre amostras — exige rigor no split da §5.1 para não virar vazamento.** Comece por (ii), que é honesta e barata, e teste (i) como melhoria medida.

Detalhe de implementação da opção (i): as sequências têm **comprimento muito variável** (50 a 216 quadros, §0), então uma janela fixa de 20 pacotes não ladrilha a repetição inteira. Defina a regra de sobra por escrito — descartar a cauda ou preencher — e mantenha a mesma regra em treino e teste.

**As 5 amostras não devem ser mediadas.** Elas são o único eixo que carrega informação de velocidade dentro de um pacote (ver §5.7): colapsar esse eixo por média joga fora exatamente o que distingue movimentos rápidos de lentos.

### 5.7 Taxa de 100 Hz e risco de aliasing de Doppler

A cota superior do desvio Doppler é `f_D = 2v/λ`, atingida na geometria **quase-monoestática** (TX e RX aproximadamente na mesma direção vista do alvo). A portadora exata do dataset **não é publicada** — o CSI-Former diz só "5 GHz". Tomando a cota:

| λ (portadora) | v acima da qual `2v/λ` passa de Nyquist (50 Hz) |
| --- | --- |
| 5,15 cm (5,825 GHz) | **1,29 m/s** |
| 6,0 cm (5,0 GHz) | **1,50 m/s** |

Com CSI a 100 Hz, movimentos de membro acima de ~1,3–1,5 m/s **podem** ultrapassar Nyquist — plausível em `throw`, `run` e `jump`. **É um envelope de pior caso, e dois fatores puxam em direções opostas:**

- **Para baixo (geometria).** O Doppler bistático real é `(2v/λ)·cos δ·cos(β/2)`, onde β é o ângulo TX–alvo–RX. Com o sujeito **sobre a linha de base** entre TX e RX, β ≈ 180° e `cos(β/2) ≈ 0`: a soma dos percursos TX→alvo→RX é estacionária em primeira ordem, e o Doppler colapsa para **qualquer** direção de movimento — é uma **zona cega geométrica**, não uma troca de componente dominante. A cota só é atingida com o alvo fora da linha de base.
- **Para cima (amplitude), mas fracamente.** Se o dado for amplitude (§4.1), o espectro de `|h(t)|` ganha termos cruzados que chegam a `4v/λ`. Porém esse termo é **móvel × móvel**, de segunda ordem, enquanto o termo em `2v/λ` é **estático × móvel**, de primeira ordem. Num ambiente indoor com linha de visada — o caso deste dataset, e a razão pela qual o próprio WiPose precisa **subtrair a média** para remover o componente estático dominante (§2.1) —, a cauda em `4v/λ` fica da ordem de **25 dB abaixo** do termo principal, provavelmente sob o piso de ruído. Trate-a como cauda de pior caso, não como o regime operante.

**Cautela com a extrapolação a partir dos artigos.** É tentador usar como conforto o fato de que o Wi-Pose degrada só de 28,3 para 29,7 mm ao cair de 1000 para 250 Hz (*"the variation of the packet rate has little effect"*) e o Winect fica abaixo de 5,8 cm a 250 pkt/s. Mas 250 pkt/s dá Nyquist de 125 Hz, o que pela fórmula desta seção corresponde a v ≈ 3,5 m/s — acima da maior parte do movimento avaliado nesses artigos, ainda que sem garantir ausência de dobra em gestos muito rápidos. Ou seja, os dois medem sobretudo **perda de resolução temporal**, não dobra espectral, e extrapolar deles para 100 Hz (Nyquist de 50 Hz, v ≈ 1,3–1,5 m/s) é atravessar exatamente a fronteira em discussão. Note também que o Winect **recomenda** 250 pkt/s como mínimo — por degradação de acurácia e encurtamento do segmento, não por aliasing — mas ainda assim a menor taxa validada na literatura é 2,5× a nossa.

**Conclusão: hipótese testável para a tarefa (e), não limitação declarada nem risco descartado.** O experimento que decide é procurar dobra espectral nas janelas agregadas de 1 s.

**Consequência prática imediata:** o *movement energy ratio* do Winect (§2.2) **não é aplicável por pacote**, e mesmo agregado ele degrada nesta taxa — ver §7, onde isso vira critério de escolha. Por pacote, o que 5 amostras sustentam é **variância e energia por subportadora** e diferenças de primeira ordem entre amostras consecutivas.

### 5.8 Pares de ações que podem ser confundidos

`sit down` × `stand up` são aproximadamente **reversos no tempo** um do outro — mesma trajetória, sentido oposto. (`push` × `pull` e `crouch` × `jump` são frequentemente citados como pares análogos, mas não são reversos de fato: têm cinemáticas de braço distintas e fases balísticas diferentes. Trate só o primeiro par como reverso genuíno.)

**Cuidado com o raciocínio fácil aqui.** É verdade que inverter o tempo inverte o sinal das diferenças de primeira ordem de qualquer série real — mas isso vale para temperatura e preço de ação também; não é evidência de que a direção do *movimento corporal* seja recuperável da amplitude. E há uma razão física para desconfiar: `|h(t)|` de um canal multipath percorre um **ciclo de desvanecimento a cada ~λ de variação de percurso** (≈5 cm), e em geometria quase-monoestática o percurso refletido varia a **2v**, não a `v`. Ou seja, a taxa de desvanecimento **é** o próprio `f_D = 2v/λ` da §5.7: um membro a 1 m/s gera **~33 a 40 ciclos por segundo**, conforme a portadora (e note que isso já fica perto do Nyquist de 50 Hz — as duas seções fecham na mesma conta).

Numa janela de 1 s a amplitude oscila algumas dezenas de vezes, e a soma das diferenças de primeira ordem **telescopa** para `(a_fim − a_início)` — uma única amostra de desvanecimento, essencialmente aleatória. **Janela maior não revela tendência; acumula mais nulos.**

Há ainda um limite estrutural: `sit down` e `stand up` são reversos na escala da **sequência inteira** (5–6 s). Dentro de um pacote de 50 ms existem só 4 diferenças, e **nenhuma estatística simétrica por pacote distingue esse par** — ela é invariante à reversão por construção. Features assimétricas (diferenças com sinal) são teoricamente admissíveis, mas devem ser fracas neste regime, pelo argumento de telescopagem acima. Ou seja: o esquema da §5.6 opção (ii) — classificar por pacote com votação — tem chance baixa neste par, e a opção (i), com janela agregada de 1 s, é o caminho provável. **Hipótese a testar, não impossibilidade demonstrada.**

> **Hipótese testável, formulada sobre o que de fato carrega direção:** `sit down` e `stand up` se separam por **envelope de energia por janela** (a energia de movimento sobe e desce em ordem diferente nos dois) ou por **taxa de cruzamentos por zero** das diferenças, e **não** pela soma das diferenças de primeira ordem. Se o `CSI` for complexo (§4.2), o **produto conjugado entre antenas RX do mesmo rádio** preserva o Doppler relativo com sinal e é de longe a feature mais forte para este par.

Registre a hipótese **antes de rodar** e teste as três features. **Desenho do teste:** não leia a célula da matriz 12×12 — com ~24 sequências por classe ela não tem resolução (§5.1.1). Faça uma **comparação binária dirigida**: classificador de duas classes sobre as ~48 sequências de `sit down` e `stand up`, com bootstrap pareado entre as features. Se o par continuar colapsado sob todas elas, isso é um resultado legítimo da tarefa (e) — e a explicação (a amplitude simplesmente não codifica direção neste regime de desvanecimento) é boa análise crítica.

### 5.9 Risco operacional: 166 mil arquivos pequenos dentro do OneDrive

A volumetria (0,9 a 3,6 GB conforme dtype, §4.1) não é problema. A **contagem de arquivos** é: são ~166.600 `.mat` HDF5 individuais, e o repositório vive em `C:\Users\...\OneDrive\Documents\...`. Abrir 166 mil arquivos pequenos num diretório sincronizado é ordens de magnitude mais lento que ler o mesmo volume em bytes contíguos, dispara sincronização na nuvem, e o *Files On-Demand* pode deixar arquivos como *stubs* que travam a leitura.

**Recomendações:**

- Extrair o `.rar` **fora do OneDrive** (ex.: `C:\datasets\wipose`) e apontar o caminho por variável de ambiente ou arquivo de config — assim cada integrante usa o caminho da própria máquina.
- No passo 4, gerar um **cache consolidado** (`.npy` ou shards HDF5) **no mesmo diretório fora do OneDrive** (`C:\datasets\wipose\cache`), apontado pela mesma variável de ambiente — não em `artifacts/`, senão os vários GB voltam para dentro da pasta sincronizada, que é o problema que este item existe para evitar. O cache passa a ser a fonte de tudo depois. Ler 166 mil arquivos uma vez, nunca mais.

**Dois ajustes na `.gitignore` que vão doer se ficarem como estão:**

- `*.mat` é global e bloqueia commitar uma **fixture pequena de teste**. Abrir exceção (ex.: `!tests/fixtures/*.mat`) para que os testes rodem na máquina de qualquer um sem o dataset completo.
- `artifacts/` é ignorado por inteiro — e é exatamente para lá que os passos 4 e 7 mandam as figuras de amplitude/fase, a animação do esqueleto e as matrizes de confusão. **Nenhuma figura que P3 precisa para o relatório ficaria versionada**, e com três pessoas em máquinas diferentes isso é atrito garantido. Só `artifacts/figuras/` deve existir no repo, e versionado — o cache pesado mora fora do OneDrive (bullet acima).

---

## 6. Estado atual do repositório

> **Nota:** esta seção descrevia o repositório **antes** da inspeção do dataset.
> O estado atual, depois do primeiro commit de código, é o de baixo.

```
AICT-I-WiPose/
├── Artigos/            ✅ dois PDFs
├── artifacts/figuras/  ✅ figuras versionadas; cache pesado fica fora do repo
├── dataset/Wi-Pose/    ✅ baixado (2,4 GB, ignorado no Git)
├── docs/               ✅ analise-inicial.md + guia-do-projeto.md
├── scripts/            ✅ inspecionar.py, plotar_amplitude.py
├── src/dados.py        ✅ o carregador (contrato único de acesso ao dataset)
├── tests/              ✅ suíte + 4 fixtures pequenas versionadas
├── tema.pdf            ✅
├── requirements.txt    ✅ corrigido (h5py obrigatório; scipy.io.loadmat não serve)
└── .gitignore          ✅ dataset fora; fixtures e figuras dentro
```

**Dataset não baixado.** Download em [Google Drive](https://drive.google.com/file/d/1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9/view?usp=sharing) ou Baidu (senha `wqeq`), conforme `dataset/README.md`. Vários GB — combinar quem baixa e se todos precisam da cópia integral (para os primeiros passos, uma amostra por ação e por faixa de `id` basta).

**O que dá para adiantar sem o dataset:** a **assinatura** da interface (`carregar_split(split, cache_dir) -> (X, y, grupos)`), a forma de saída `(9, 30, 5)`, o esqueleto do código de P2 contra dados sintéticos, e toda a tarefa (a). **O que não dá:** os outros cinco itens do contrato (§8) — leitor, dtype complexo/real, ordenação de `SkeletonPoints`, regra de split e critério de subportadora ruidosa — dependem dos passos 1 e 2, e é por isso que a §9 separa o contrato em 3a e 3b.

⚠️ **O download é ponto único de falha para as tarefas (b), (c), (d) e (e) ao mesmo tempo, e as duas fontes são frágeis:** um único link do Google Drive de vários GB (sujeito a "quota exceeded" em arquivos populares) e o Baidu Netdisk (que na prática exige cliente e conta, impraticável fora da China). **Plano B, se o dataset não chegar na primeira semana:** abrir issue no repositório do NjtechCVLab, e — persistindo — rescopar para um dataset público equivalente (MM-Fi ou Person-in-WiFi-3D, ambos suportados pelo mesmo `DT-Pose`). A tarefa (a) e toda a análise dataset × artigo da §3 se preservam intactas nesse rescopo.

**Ambiente:** `.venv/` já existe, mas o `requirements.txt` precisa dos ajustes do passo 0 (§9) antes de servir. Há GPU disponível (RTX 5060, 8 GB) — mas o código deve selecionar o dispositivo com `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, nunca cravando `"cuda"`, senão só roda na máquina de quem tem placa.

---

## 7. Caminhos possíveis

Três rotas, da mais segura à mais ambiciosa. **Não são mutuamente exclusivas** — a Rota A é pré-requisito das outras duas.

### Rota A — Classificação de ações (o caminho que a disciplina pede)

**Escopo:** exatamente as tarefas (b) → (d) → (c) → (e).

- **Alvo:** rótulo de ação, 12 classes, extraído do nome do arquivo.
- **Baselines:** **taxa de acaso = 1/12 ≈ 8,3%** e **taxa da classe majoritária**, que é a frequência da classe mais frequente e será **maior que 8,3%** — o dataset não é perfeitamente balanceado (a Figura 3 do CSI-Former mostra contagens diferentes por ação). O número exato só depois do passo 1.
- **Métricas:** **acurácia balanceada / F1-macro** como número de manchete (com 12 classes desbalanceadas, a acurácia simples esconde classe minoritária colapsada), mais **recall por classe** ao lado da matriz de confusão. A tarefa (e) é justamente sobre a *estrutura* do erro.
- **Entrada do modelo:** o tensor `(9, 30, 5)` — antenas, subportadoras, tempo — **mantendo o eixo temporal de 5**. Escolhido **por mérito**, não por consenso: manter antena e subportadora em eixos separados é pré-requisito da tarefa (e). Coincide com 2 dos 3 carregadores públicos, mas não há convenção estabelecida (§4.1). Uma CNN 2D sobre `(30, 5)` com 9 canais, ou Conv3D. Qualquer agregação temporal tem que ser declarada explicitamente, nunca implícita num `reshape`.
- **Modelos:** começar por SVM/Random Forest sobre features estatísticas de amplitude — **média, variância e diferenças de primeira ordem entre as 5 amostras**, por subportadora e por par de antenas. As diferenças de primeira ordem capturam a **dinâmica intra-pacote**; note que elas **não** resolvem os pares reversos da §5.8, que exigem features de janela longa. (Curtose sobre n=5 não estima nada; se quiser momento de ordem alta, calcule **através** das 30 subportadoras ou dos 9 pares, onde n é grande o bastante.) Depois, uma CNN pequena. O `tema.pdf` pede "modelo simples" — cumprir isso à risca e só depois escalar.
- **Critério de "subportadora ruidosa"** — a tarefa (e) exige um, e ele precisa ser fixado no passo 3, antes da ablação. **Recomendação: variância temporal ou informação mútua com o rótulo**, não o *movement energy ratio* do Winect. Motivo: a razão do Winect é `energia(0–20 Hz) / energia total`; a 1000 pkt/s o denominador cobre 0–500 Hz e a banda de movimento é ~4% do espectro, o que dá faixa dinâmica de sobra. A **100 Hz o denominador cobre só 0–50 Hz e a banda vira 40%** — a faixa dinâmica comprime de ~25× para ~2,5×, reduzindo muito a separabilidade. Pior: pela conta da §5.7, a banda 0–20 Hz corresponde a v ≲ 0,5 m/s, então a 100 Hz a razão deixa de ser detector de **movimento de membro** e vira detector de **movimento lento** — que é outra coisa. Além disso, o Winect a usa para escolher **uma** subportadora por segmento de 0,1 s, recalculada dinamicamente; nunca a valida como ranking global estável, que é o que uma ablação exige.
  Antes de fixar: **medir a distribuição do critério escolhido sobre as 30 subportadoras** (se estiver comprimida, não serve) e **verificar se o ranking é estável entre ações e sujeitos** — essa checagem de estabilidade é, por si só, um resultado da tarefa (e). Com o critério fixo, a ablação vira uma curva com significado: **remover as k de menor pontuação × remover as k de maior**, comparando as duas curvas, com passos de `k` grandes o bastante para superar o IC **da diferença pareada** (§5.1.1) — tipicamente ~1,5 a 2 pp, não os ±6 pp do valor absoluto.
- **Visualização.** O `tema.pdf` pede visualizar os movimentos *"a partir das variações do sinal de rádio"* — então animar `SkeletonPoints` **não fecha o requisito**: isso visualiza a anotação armazenada, sem passar pelo CSI. Faça as duas coisas, com papéis distintos: (i) a animação do esqueleto 2D como **inspeção da qualidade da anotação** e teste da ordenação dos 54 valores (§4.1, §5.5) — custa quase nada; e (ii) para o requisito de fato, uma visualização **derivada do rádio** — espectrograma ou energia por subportadora ao longo da repetição, sincronizado com a ação predita. Se a Rota B não for executada, dizer no relatório que "visualizar" foi atendido no nível do **sinal**, não da pose reconstruída.
- **Veredito:** ✅ **é esta que fazemos.** Entrega as cinco tarefas, tem baseline claro e produz números defensáveis.

### Rota B — Regressão de pose 2D

**Escopo:** prever as **36 coordenadas** (18 juntas × `(x, y)`) a partir do CSI, replicando parcialmente o CSI-Former.

- **Métrica:** PCK@k. ⚠️ **O 0,8417 publicado não é comparável a um número nosso:** o split deles não é declarado — embora a medição mostre que ele agrupa por sequência (§4.1), não sabemos se foi assim que o modelo deles foi treinado e avaliado. Além disso os limiares 5 e 50 do PCK@k só fazem sentido em **pixels**, o que torna a resolução/unidade das coordenadas (§4.2, pergunta 6) pré-requisito de qualquer comparação — confirme a definição exata no CSI-Former antes de comparar. **Se quiserem comparabilidade real:** treinar também com o split original `Train/`/`Test/` e reportar os dois lado a lado — esse par de números é, aliás, o experimento mais interessante da tarefa (e).
- **Riscos:** ordem de magnitude mais caro; ground truth ele mesmo ruidoso; e **não é o que a tarefa (c) pede**.
- **Veredito:** ⚠️ **extensão opcional**, só depois da Rota A fechada. Rende um "trabalhos futuros" forte no relatório mesmo sem ser executada.

### Rota C — Reprodução do WiPose/3DVP

**Escopo:** cinemática direta com quatérnions ou perfil de velocidade 3D.

- **Riscos:** o dataset **não sustenta**, por três razões independentes e verificadas: sem ground truth 3D; sem a geometria das antenas que a Eq. 11 exige; 50 ms de janela não bastam para a STFT do perfil Doppler. (Há ainda o risco de aliasing da §5.7, mas esse é hipótese não testada e não entra na conta das três razões.). Quanto a rodar o método no dado original: **não localizamos release público** dos dados do MobiCom — mas isso é afirmação negativa e não foi testada com os autores; as três razões acima já fecham a rota com o dataset que temos.
- **Veredito:** ❌ **não executável com este dado.** Entra no relatório como discussão teórica na tarefa (a) — descrever o método e explicar *por que* ele não se aplica aqui é, em si, uma contribuição analítica.

---

## 8. Divisão sugerida entre as 3 pessoas

A cadeia (b) → (d) → (c) → (e) é sequencial, então uma divisão ingênua deixaria duas pessoas ociosas. A saída é: **todo mundo escreve sobre um contrato de dados combinado no dia 1**, e (a) roda em paralelo do começo ao fim.

| Pessoa | Frente principal | Entrega |
| --- | --- | --- |
| **P1 — Dados e DSP** | `ED` + `PC` | `src/dados.py` (loader dos `.mat`, parsing do nome de arquivo para ação/sequência, cache consolidado, splits) e `src/dsp.py`. **Filtragem temporal:** Hampel, Butterworth, wavelet, e — se houver fase — unwrap + sanitização linear. **Filtragem espacial** (o `tema.pdf` pede as duas, e a lista temporal não cobre isso): combinação/média ponderada entre os 9 pares de antena, PCA sobre o eixo de antenas, e o produto conjugado entre antenas do mesmo rádio. Figuras por antena e subportadora. |
| **P2 — Modelagem** | `CM` | `src/modelos.py` + `src/treino.py`. Baselines triviais → SVM/RF sobre features → CNN pequena. Seed fixa, split conforme §5.1, normalização ajustada só no treino, teste aberto uma vez só. |
| **P3 — Bibliografia e análise** | `RB` + `AC` | Seções 2 e 3 do relatório (contraste Wi-Pose × Winect com a ressalva das juntas não comparáveis; discrepância dataset × artigo). Depois: matrizes de confusão, ablação de subportadoras, curva acurácia × nº de subportadoras, comparação entre os splits. |

**O contrato que trava tudo (definir no dia 1, juntos).** A interface principal é **em lote**, não arquivo a arquivo — `carregar_split(split, cache_dir) -> (X, y, grupos)` —, porque a §5.9 já condena ler 166 mil `.mat` a cada execução; `carregar_pacote(caminho)` fica como detalhe interno usado uma vez para construir o cache. Se P2 escrever o treino contra a leitura arquivo a arquivo, todo o código de acesso a dados é refeito quando o cache chegar — exatamente o retrabalho que o contrato existe para evitar.

O contrato fixa explicitamente:

- **o leitor** (`h5py` ou `mat73`) — eles dão ordens de eixo diferentes para o mesmo arquivo;
- **a forma de saída**: `(9, 30, 5)`, **com a convenção do eixo 9 declarada** — qual índice é TX e qual é RX (ex.: `tx = idx // 3`, `rx = idx % 3`), sem a qual as prescrições de produto conjugado das §5.2 e §5.8 não são implementáveis;
- **se `csi` é complexo ou amplitude** (§4.2, pergunta 1) — isso define se as tarefas (b) e (d) têm fase;
- **a ordenação de `SkeletonPoints`** (§4.2, pergunta 4), validada visualmente;
- **a regra de split**, de onde sai a validação, e o vetor `grupos` que a materializa (§5.1);
- **a família do critério de subportadora ruidosa** (§7) — decidida pelos três, executada por P3; o valor final só se fixa depois de medi-lo sobre treino/validação (passo 4b da §9).

**Uma regra de código que entra no contrato:** proibido converter CSI para real com `astype`. Num array complexo do NumPy, `.astype(np.float32)` **não levanta erro** — emite um `ComplexWarning` e descarta a parte imaginária silenciosamente. Se o dado for complexo e alguém escrever esse idioma, a fase que a tarefa (d) inteira precisa desaparece sem sinal nenhum. A conversão para amplitude é sempre `np.abs()` explícito.

Com esse contrato fixo, P2 escreve o modelo contra dados sintéticos de forma correta enquanto P1 ainda resolve o `.mat`, e P3 já monta o esqueleto do relatório.

**Rodízio no fim:** P3 assume a análise crítica só quando P2 tiver resultados — até lá, (a) ocupa a pessoa integralmente e ainda sobra para ajudar P1 nas figuras.

---

## 9. Próximos passos concretos

| # | Ação | Responsável | Bloqueia |
| --- | --- | --- | --- |
| ~~0~~ | ✅ **Feito.** Promessas de "3D" removidas do `README.md` e do `requirements.txt`; comentário do `scipy` corrigido; `.gitignore` ajustado; `dataset/README.md` reescrito com a estrutura real. **Decisão registrada: o leitor é `h5py`, `mat73` foi dispensado** — ele restaura a ordem de eixos do MATLAB e reabriria a divergência que o contrato existe para fechar. | — | — |
| ~~1~~ | ✅ **Feito.** Dataset baixado e verificado: 120 ids por ação ⇒ o campo é **sequência**, não voluntário. | — | — |
| ~~2~~ | ✅ **Feito** — `scripts/inspecionar.py`. Chaves `CSI`/`SkeletonPoints`, v7.3, CSI é amplitude real, esqueleto em blocos com **y antes de x**. | — | — |
| ~~3a~~ | ✅ **Feito** — `src/dados.py`. Leitor `h5py`, saída `(9,30,5)` com `tx*3+rx`, juntas `(18,3)` como `(x,y,c)`. | — | — |
| 3b | Fechar a **regra de split**, o campo `grupos` e escolher a **família** do critério de subportadora (variância temporal ou informação mútua) — depende do passo 1 (não dá para fixar `sequencia` sem saber o que o `id` significa). O valor do critério só se fixa no 4b, quando houver dado. | Todos | 4b, 5, 7 |
| 4 | 🟡 **Parcial** — `src/dados.py` e `scripts/plotar_amplitude.py` prontos (figuras de amplitude por subportadora, evolução temporal, comparativo das 12 ações). **Falta:** cache consolidado e a animação do esqueleto. A contingência da fase se confirmou: não há fase, e a ausência está documentada. | P1 | 5, 6 |
| 4b | Medir a distribuição e a **estabilidade entre ações/sujeitos** do critério de subportadora escolhido em 3b — **só sobre treino/validação**, nunca no teste (§5.1.1) → fixa o critério | P3 | 7 |
| 5 | Baselines (acaso, classe majoritária) + SVM sobre features de amplitude, **com IC por bootstrap sobre sequências** (§5.1.1) → primeiro número honesto | P2 | 5b, 6, 7 |
| 5b | CNN pequena sobre `(9,30,5)` + comparação com o SVM **na validação**; escolha do modelo final → **fecha `CM`** | P2 | 7 |
| 6 | Filtragem e — **se houver fase** (§4.2) — sanitização, medidas **na validação** contra o baseline de amplitude → fecha `PC` | P1 + P2 | 7 |
| 7 | Matriz de confusão + recall por classe **com IC** + ablação de subportadoras (critério fixado em 4b, **decidida na validação**, com bootstrap pareado) + comparação entre splits + teste dirigido das hipóteses de §5.7 e §5.8 + **espectrograma/energia por subportadora sincronizado com a ação predita** (é esta figura, e não a animação do esqueleto, que fecha o requisito de "visualizar" do `tema.pdf`) → fecha `AC` | P3 | Relatório |
| 8 | Revisão bibliográfica escrita → fecha `RB` | P3 | — (paralelo desde já) |

**Duas decisões que dependem só dos passos 1 e 2, e que mudam o semestre inteiro:**

1. **O `id` decodifica em voluntário?** Se não, o split por sujeito cai e passamos ao split por sequência — com a ressalva explícita no relatório de que o número é intra-sujeito.
2. **O `CSI` é complexo?** Se não, a tarefa (d) se rescopa: sem fase para calibrar, ela vira filtragem temporal e seleção espacial/espectral. Ainda é tarefa (d) legítima, mas metade do plano de `src/dsp.py` muda.

Melhor descobrir as duas agora do que na véspera da entrega.

---

## 10. Referências

- Jiang, W., Xue, H., Miao, C., Wang, S., Lin, S., Tian, C., Murali, S., Hu, H., Sun, Z., Su, L. **"Towards 3D Human Pose Construction Using WiFi."** *MobiCom '20*. [doi:10.1145/3372224.3380900](https://doi.org/10.1145/3372224.3380900)
- Ren, Y., Wang, Z., Tan, S., Chen, Y., Yang, J. **"Winect: 3D Human Pose Tracking for Free-form Activity Using Commodity WiFi."** *IMWUT* 5(4), art. 176, 2021. [doi:10.1145/3494973](https://doi.org/10.1145/3494973)
- Zhou, Y. et al. **"CSI-Former: Pay More Attention to Pose Estimation with WiFi."** *Entropy* 25(1):20, 2023. [mdpi.com/1099-4300/25/1/20](https://www.mdpi.com/1099-4300/25/1/20) — artigo que publica e usa o dataset que efetivamente adotamos.
- Fang, H. et al. **"RMPE: Regional Multi-person Pose Estimation."** *ICCV 2017* — AlphaPose, origem das anotações.
- Dataset: [NjtechCVLab/Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset)
