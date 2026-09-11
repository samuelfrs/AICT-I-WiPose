# Análise: migração do Wi-Pose para o MM-Fi

**Código da tarefa:** `RB-02`
**Data:** 10/09/2026

> Números marcados ✅ foram verificados abrindo os arquivos do MM-Fi nesta máquina
> — quando o dado vem de amostra, o tamanho da amostra está dito. Números do
> artigo (NeurIPS 2023) estão identificados como tal.

---

## 1. Resumo: a troca resolve os dois problemas que travavam o projeto

O Wi-Pose tinha duas limitações que quebravam metade do enunciado: não havia
**fase** no sinal (tarefas b e d), e as anotações eram **2D** quando o `tema.pdf`
fala em 3D. O MM-Fi resolve as duas.

| | Wi-Pose (antigo) | **MM-Fi (novo)** |
| --- | --- | --- |
| Anotação de pose | 2D, 18 juntas (AlphaPose) | ✅ **3D, 17 juntas**, em metros |
| Fase do CSI | ❌ não distribuída | ✅ **`CSIphase`**, crua e envolvida em ±π |
| Subportadoras | 30 | ✅ **114** |
| Ações | 12 | ✅ **27** |
| Sujeitos | 12 (não identificáveis) | ✅ **40, identificados** (`S01`–`S40`) |
| Ambientes | 1 | ✅ **4** (`E01`–`E04`) |
| Split por pessoa | ❌ impossível | ✅ **protocolo oficial** (32 treino / 8 teste) |
| Split por ambiente | ❌ inexistente | ✅ **protocolo oficial** (3 treino / 1 teste) |
| Quadros por sequência | irregular (50–216) | ✅ **297, uniforme** |
| Quadros de CSI duplicados | ~10,6% de preenchimento | ✅ **zero** |
| Baseline publicado | nenhum | ✅ MPJPE de pose 3D por protocolo |

⚠️ Repare que o **número de sequências cai** — 1.440 no Wi-Pose contra 1.080 aqui.
O ganho é em uniformidade, número de sujeitos e total de quadros (320.760), não em
quantidade de sequências.

**Conclusão: a migração vale a pena.** Mas o MM-Fi tem defeitos próprios, e dois
deles são sérios — §2.5 e §2.6.

---

## 2. O que tem no dataset (verificado)

```
MMFi_Dataset/
└── E01..E04/          4 ambientes
    └── S01..S40/      40 sujeitos (10 por ambiente)
        └── A01..A27/  27 ações
            ├── ground_truth.npy    pose 3D da sequência inteira (297, 17, 3)
            ├── wifi-csi/           frame001.mat ... frame297.mat
            ├── mmwave/  lidar/  rgb/  depth/  infra1/  infra2/
```

✅ **1.080 sequências, todas completas**, **297 quadros cada**, **320.760 quadros**.
✅ Perfeitamente balanceado: 40 sequências por ação.

⚠️ **Cada sujeito aparece em exatamente um ambiente** (S01–S10 só em E01, e assim
por diante — verificado nos 40). Ou seja, **S2 e S3 não são eixos independentes**:
qualquer conjunto de 8 sujeitos retidos é também um conjunto parcial de ambiente
retido. Vale dizer isso no relatório em vez de tratá-los como testes ortogonais.

### 2.1 O sinal — `wifi-csi/frameNNN.mat`

Formato **MATLAB v5**, então `scipy.io.loadmat` abre normalmente (o Wi-Pose era
v7.3/HDF5 e exigia `h5py`). Duas variáveis, ambas `(3, 114, 10)` float64 =
**(3 antenas RX, 114 subportadoras, 10 amostras temporais)**:

| Variável | Faixa medida |
| --- | --- |
| `CSIamp` | ✅ **0,00 a 52,6** (amostra de ~430 arquivos, 4 ambientes) |
| `CSIphase` | ✅ `[-3,13, +3,1416]` — envolvida em ±π |

Cada arquivo cobre **100 ms** — o dobro da janela do Wi-Pose.

Hardware (artigo): TP-Link N750 + Atheros CSI Tool, **5 GHz, 40 MHz de banda**,
1 antena TX e 3 RX, CSI a ~100 Hz, sincronizado com as outras modalidades a 10 Hz
com erro < 25 ms.

*Sobre a escala da amplitude:* a faixa e a presença de `-inf` (§2.6) sugerem
**escala logarítmica (dB)**, mas **o artigo não declara isso** — trate como
hipótese a confirmar, não como fato.

### 2.2 ⚠️ A fase é crua — a tarefa (d) tem objeto de novo

Verificado sobre **2.400 traços** (antena × amostra temporal, vários sujeitos):

- A faixa bate em **±π exatos** → fase **envolvida** (*wrapped*).
- Após `unwrap` ao longo das subportadoras, sobra uma **componente linear** — o
  offset por atraso de detecção de pacote, que o artigo Wi-Pose chama de `ε(f,t)`.

**Mas não é tão limpo quanto parece.** A inclinação varia de **−0,10 a +0,23
rad/subportadora**, e **14,2% das inclinações são negativas**. Mais importante:
a rampa linear **frequentemente não domina** — o |R| mediano do ajuste é 0,837, e
**71,4% dos traços ficam com |R| < 0,9**.

**Leitura honesta:** a regressão linear por subportadora é o *ponto de partida*
da tarefa (d), não a resposta pronta. Vale testar também a **multiplicação
conjugada entre antenas do mesmo receptor**, que cancela os offsets comuns sem
supor que a rampa seja linear e dominante.

*Ressalva adicional:* o artigo (§3.1) diz que o CSI passa por *"average sliding
window method inside the firmware"* e que a matriz 3×114×10 é **aumentada**. Ou
seja, já há suavização embutida, e as 10 amostras podem não ser 10 instantâneos
independentes.

### 2.3 ⚠️ O ground truth 3D não é COCO — é Human3.6M

O artigo diz "17 keypoints" e é fácil assumir COCO. **Não é.** Verificado sobre 8
sujeitos de 4 ambientes:

| Teste | Se fosse COCO-17 | Como Human3.6M-17 |
| --- | --- | --- |
| Comprimento cabeça→pés | 0,32 m (absurdo) | ✅ **1,45 m** |
| Simetria dos ombros em y | 0,372 | ✅ **0,015** |

Ordem das 17 juntas (Human3.6M):

```
 0 quadril (raiz)   1 quadrilD   2 joelhoD   3 péD
 4 quadrilE         5 joelhoE    6 péE       7 coluna
 8 tórax            9 pescoço   10 cabeça
11 ombroE          12 cotoveloE 13 punhoE
14 ombroD          15 cotoveloD 16 punhoD
```

Dois detalhes que quebram em silêncio:

- ✅ **Unidade: metros.** O artigo reporta MPJPE em **mm** — é ×1000.
- ✅ **O eixo `y` cresce para BAIXO.** Cabeça em `y ≈ -0,70`, pés em `y ≈ +0,77`.
  Plotar sem inverter desenha a pessoa de cabeça para baixo.

Origem: centro da câmera RGB. O `z` fica em torno de 3,0 m — distância até a câmera.

*(O vão cabeça→pés de 1,45 m é do esqueleto anotado, não a estatura das pessoas:
o artigo declara **172,3 ± 7,9 cm**. O esqueleto Human3.6M não vai do topo da
cabeça até a sola do pé.)*

### 2.4 ⚠️ O "ground truth" é derivado de modelo, não de captura de movimento

Diferente do Wi-Pose original (que usava VICON, erro < 2 mm), aqui a anotação 3D
vem de um pipeline: **HRNet-w48** estima pose 2D em duas câmeras infravermelho →
triangulação → otimização com regularizadores. Qualidade declarada: **PCKh@0,5 =
95,66%**.

É bom, mas é **saída de modelo**. Qualquer MPJPE que a gente reportar é erro
*contra outra estimativa*, não contra a verdade física. Isso precisa estar no
relatório.

### 2.5 🚨 75% das sequências têm juntas congeladas no ground truth

Este é o achado mais sério, e ele **não é ruído — é deliberado**.

Verificado nas **1.080 sequências** (uma junta "congelada" = as 297 posições são
bit-idênticas):

| Juntas congeladas | Sequências |
| --- | --- |
| 0 | 271 |
| 1–3 | 449 |
| 7 | 80 |
| 12–13 | 200 |
| **15 de 17** | **80** |

- ✅ **809 de 1.080 sequências (74,9%)** têm ao menos uma junta congelada.
- ✅ **280 sequências têm 10 ou mais** juntas congeladas.
- ✅ Em **A07 e A08** (extensão de membro), **15 das 17 juntas** ficam paradas a
  sequência inteira.
- A distribuição é múltipla de 40 (= nº de sujeitos) → é **determinístico por
  ação**, não acidente.

A causa está no artigo (§4.3): *"we set limitations on some joints based on the
actions and bone lengths, which is why we introduce the action regularizer L_A."*
Ou seja, o pipeline de anotação **fixa** juntas que a ação não deveria mover.

**Por que isso importa muito para a Rota B (pose 3D):** um modelo que simplesmente
aprenda a emitir um *template* estático por ação obtém MPJPE aparentemente
razoável sem aprender nada sobre o sinal. A métrica fica inauditável. Quem for por
esse caminho precisa reportar o erro **apenas nas juntas que de fato se movem**, ou
o número não significa nada.

### 2.6 ⚠️ Existem valores `-inf` na amplitude

✅ Verificado em **10.692 arquivos** (4 ambientes, 12 sujeitos, 36 sequências):

- **322 arquivos (3,01%)** contêm valor não-finito em `CSIamp`.
- **29 das 36 sequências** amostradas são afetadas.
- `CSIphase`: **zero** ocorrências.

Exemplos: `E01/S10/A03/frame042.mat`, `E01/S10/A19/frame028.mat`.

**Consequência prática:** sem tratamento, qualquer média, desvio padrão, SVM ou
CNN produz `NaN` silenciosamente e o treino vira lixo sem erro nenhum. O
carregador precisa lidar com isso explicitamente (substituir, ou descartar o
quadro e registrar quantos foram).

### 2.7 As 27 ações

**14 atividades diárias** (P1) — executadas livremente no espaço:
`A02` expansão do peito (horizontal) · `A03` expansão do peito (vertical) ·
`A04` torção (esquerda) · `A05` torção (direita) · `A13` levantar mão esquerda ·
`A14` levantar mão direita · `A17` acenar esquerda · `A18` acenar direita ·
`A19` pegar objeto · `A20` arremessar (esquerda) · `A21` arremessar (direita) ·
`A22` chutar (esquerda) · `A23` chutar (direita) · `A27` curvar-se

**13 exercícios de reabilitação** (P2) — em posição fixa:
`A01` alongar e relaxar · `A06` marcha estacionária · `A07` extensão de membro
(esquerda) · `A08` extensão (direita) · `A09` afundo (frente-esquerda) ·
`A10` afundo (frente-direita) · `A11` extensão (ambos) · `A12` agachamento ·
`A15` afundo (esquerda) · `A16` afundo (direita) · `A24` extensão do corpo
(esquerda) · `A25` extensão do corpo (direita) · `A26` salto

⚠️ **Cuidado com os rótulos "esquerda/direita".** Vários são **contralaterais** —
o artigo define A22 como *"kicking toward left direction using right leg"*. Verifiquei
no ground truth: em **A22 é o pé direito que se move** (deslocamento 0,49 m; o
esquerdo fica em 0,00), e em A23 é o esquerdo. O mesmo vale para A20/A21 (braços).

Isso **inverte a intuição** sobre quais pares são difíceis: quando o membro que se
move é diferente, a assinatura de multipath muda bastante, e a distinção pode ser
mais fácil do que parece. Não presuma — meça.

---

## 3. Os protocolos oficiais

| Split | O que mede | Divisão |
| --- | --- | --- |
| **S1** | Desempenho no domínio conhecido | Aleatório, 3:1 |
| **S2** | Generalização para **pessoa nova** | 32 sujeitos treino / 8 teste |
| **S3** | Generalização para **ambiente novo** | 3 ambientes treino / 1 teste |

Combinados com P1 / P2 / P3, dão 9 configurações.

### 🚨 O S1 **não** protege contra vazamento

O S1 é sorteio aleatório **de amostras**, não de sequências. Medi a correlação de
`CSIamp`: quadros adjacentes da mesma sequência dão **0,652**; quadros de
sequências diferentes dão **0,127**. Um sorteio por quadro coloca vizinhos a
100 ms de distância nos dois lados da divisão — exatamente o vazamento que
dominou a análise do Wi-Pose.

**Regra para o grupo:** mesmo usando o S1, **agrupe por sequência**. E prefira o
S2 como número de manchete, porque ele é o único que mede generalização de
verdade.

**Para o relatório:** reportar S1 e S2 lado a lado. A queda entre os dois mede
quanto o modelo aprendeu a pessoa em vez do movimento — é análise crítica da
tarefa (e).

---

## 4. ⚠️ Um número do artigo que vai confundir o grupo

O artigo reporta, no apêndice, reconhecimento de ação com **5,45% (AGCN)** e
**5,12% (CTRGCN)** usando WiFi. Com 27 classes, o acaso é 3,7%. Parece que o
projeto é inviável.

**O experimento deles é de duas etapas** (apêndice A.8: *"we leverage the results
obtained by human pose estimation... which serve as the dataset for skeleton-based
action recognition"*):

```
CSI  →  [modelo de pose]  →  esqueleto 3D estimado  →  [AGCN]  →  ação
```

**Mas cuidado com a explicação fácil.** Não dá para dizer que "a cascata é
frágil", porque as outras modalidades atravessam *a mesma cascata* com resultados
muito melhores — na mesma tabela do apêndice: **mmWave 65,25%, LiDAR 54,44%,
RGBD 87,78%**. A explicação do artigo é outra: *"some frames of joints predicted
by WiFi are not robustly consistent."*

**A leitura honesta é mais dura:** o esqueleto 3D **estimado a partir do WiFi**
carrega pouca informação de ação. Isso é um resultado real sobre a Rota B.

**O que isso não diz:** o artigo **não avalia classificação direta de CSI → ação
em lugar nenhum** (verifiquei nas 20 páginas). Esse é um caminho diferente, bem
estabelecido na literatura de WiFi sensing, e é o que a tarefa (c) pede:

```
CSI  →  [classificador]  →  ação
```

Sem etapa intermediária, sem erro acumulado. Os 5% **não são evidência contra a
Rota A** — mas também não são prova a favor. Ninguém publicou esse número neste
dataset, então vamos ter que descobrir.

*Nota de escala: esse benchmark tem só **162 clipes de teste** (6 por classe), e o
apêndice usa um split de **30/10 sujeitos** que contradiz os 32/8 do texto
principal. Vocês vão esbarrar nessa inconsistência ao implementar o S2.*

---

## 5. Baselines publicados (estimação de pose 3D)

Do artigo, MPJPE / PA-MPJPE em **mm**, modelo MetaFi++ (CNN + transformers):

| Split | P1 (diárias) | P2 (reabilitação) | P3 (todas) |
| --- | --- | --- | --- |
| **S1** aleatório | 186,9 / 120,7 | 213,5 / 121,4 | 197,1 / 121,2 |
| **S2** sujeito novo | 222,3 / 125,4 | 247,0 / 122,7 | 231,1 / 124,0 |
| **S3** ambiente novo | 367,8 / 121,0 | 360,2 / 117,2 | 369,5 / 116,0 |

Comparação em S1/P3 com as outras modalidades: LiDAR **92,5** / 61,5 · mmWave
**117,0** / 57,3 · RGB **279,0** / 81,2 · WiFi **197,1** / 121,2.

Duas leituras, corrigidas:

1. **O WiFi não é o pior em MPJPE** — o RGB é (279,0 contra 197,1), por causa do
   *domain gap* do modelo pré-treinado. O WiFi é o pior em **PA-MPJPE** (121,2),
   que é a métrica que isola a qualidade da pose. *(E o apêndice mostra um RGB
   re-treinado chegando a 60,5 mm, o que muda o enquadramento.)*
2. **A explosão do S3** (197 → 370 mm) com PA-MPJPE **estável** (121 → 116) é o
   achado mais interessante da tabela. Note que o MPJPE aqui já é **alinhado pela
   pelve** (artigo §5.1), então isso **não** é erro de posição global — é erro de
   **rotação e escala** do corpo. Ao trocar de ambiente, o modelo continua
   produzindo um corpo com proporções plausíveis, mas orientado errado.

**Expectativa calibrada:** estimação de pose 3D via WiFi é genuinamente difícil.
Se o grupo for pela Rota B, o resultado esperado é modesto, e isso precisa ser
enquadrado como propriedade do problema, não como fracasso do trabalho.

---

## 6. O que muda nas cinco tarefas

| | Tarefa | Com o Wi-Pose | Com o MM-Fi |
| --- | --- | --- | --- |
| **a** | Revisão bibliográfica | Wi-Pose × Winect | Igual, **mais** a comparação entre datasets e a justificativa da troca |
| **b** | Carregar e plotar | Só amplitude | ✅ **Amplitude E fase**, 114 subportadoras, 3 antenas |
| **c** | Classificação | 12 ações, intra-sujeito | ✅ **27 ações**, com S1/S2/S3 oficiais |
| **d** | Pré-processamento | Sem fase para calibrar | ✅ **Sanitização de fase de verdade** + filtragem de amplitude |
| **e** | Análise crítica | Confusão + subportadoras | ✅ Idem, **mais** a queda S1→S2→S3 e os pares contralaterais |

**Nenhuma tarefa fica sem dado.** Era o problema do Wi-Pose, e acabou.

---

## 7. Caminhos possíveis

### Rota A — Classificação das 27 ações ✅ recomendada

CSI → ação, direto, sem etapa intermediária. É o que a tarefa (c) pede.

- **Entrada:** `CSIamp` e/ou `CSIphase` sanitizada, `(3, 114, 10)` por quadro.
- **Baselines:** acaso = 3,7%; classe majoritária = 3,7% (balanceado).
- **Modelos:** SVM/Random Forest sobre estatísticas por subportadora; depois uma
  CNN pequena. O `tema.pdf` pede "modelo simples".
- **Protocolo:** S1 (agrupado por sequência) e S2 no mínimo.
- **⚠️ Decidir antes de começar: a unidade de classificação é o quadro ou a
  sequência?** Isso muda tudo:
  - *Por quadro:* 320.760 amostras, mas correlacionadas — exige agrupamento no
    split e o número é por quadro, não por ação executada.
  - *Por sequência:* só **40 por classe**, e no teste do S2 sobram **8 por
    classe**. Nessa escala o intervalo de confiança é largo e **todo número
    precisa vir com IC** (bootstrap sobre sequências).
- **Risco:** moderado. O dado é melhor que o do Wi-Pose, mas são 27 classes e não
  há baseline publicado para comparar.

### Rota B — Estimação de pose 3D ⚠️ só como extensão

Regredir as 17 juntas 3D a partir do CSI, comparando com a tabela da §5.

- **Atrativo:** é o que os artigos Wi-Pose e Winect fazem, e fecha um arco com a
  tarefa (a). Métricas comparáveis com a literatura.
- **🚨 Problema sério:** as juntas congeladas (§2.5). Um modelo que memorize um
  template por ação pontua bem sem aprender nada. **Se for por aqui, reporte o
  MPJPE restrito às juntas que se movem** — senão o número é vazio.
- **Risco:** alto custo, resultado esperado modesto, e **não é o que a tarefa (c)
  pede**. Só depois da Rota A fechada.

### Rota C — Multimodal ❌ fora de escopo

O MM-Fi tem LiDAR, mmWave, RGB e profundidade sincronizados, mas o `tema.pdf` é
sobre **sinais WiFi**. Vale no máximo uma menção comparativa na tarefa (a).

---

## 8. Custos e riscos práticos

| | |
| --- | --- |
| **Tamanho** | ✅ `wifi-csi` ≈ **16 GB**; `ground_truth` ≈ 63 MB. As outras modalidades são muito maiores e **não precisamos delas**. |
| **Contagem de arquivos** | 320.760 `.mat` de CSI. Ler um a um a cada execução é inviável — vai precisar de cache consolidado. |
| **Onde guardar** | ✅ Já está fora do OneDrive (`~/Downloads/datas/`). **Manter assim.** |
| **Formato** | ✅ MATLAB v5 — `scipy.io.loadmat` resolve; `h5py` não é mais necessário. |
| **Valores `-inf`** | ⚠️ 3% dos arquivos (§2.6). Tratar no carregador, ou o treino vira `NaN` em silêncio. |
| **Código atual** | ⚠️ Todo o `src/` e `scripts/` do repositório é específico do Wi-Pose e **não funciona** aqui. Formato, forma do tensor e nomes de variável são todos diferentes. |

---

## 9. O que precisa ser decidido pelo grupo

1. **Confirmar a Rota A** como escopo principal.
2. **Quais protocolos reportar** — sugestão: P3 com S1 (agrupado) e S2.
3. **A unidade de classificação**: quadro ou sequência (§7, Rota A).
4. **O que fazer com o código atual** do repositório, que é do Wi-Pose: apagar ou
   arquivar como registro da primeira fase.
5. **Se a Rota B entra** como extensão ou só como "trabalhos futuros".
6. **Avisar o professor da troca** e do porquê — o enunciado aponta o Wi-Pose, e a
   justificativa (2D sem fase vs 3D com fase) é forte e documentada.

---

## 10. Hipóteses para a tarefa (e) — a testar, não a presumir

Registrando **antes** de rodar, para não virar viés de confirmação. **Resultado
negativo em qualquer uma delas é entrega válida**, desde que bem medido.

1. **Pares contralaterais** (A20/A21, A22/A23, A07/A08, …): são mais fáceis ou
   mais difíceis que a média? O membro que se move é diferente, o que *poderia*
   facilitar — mas a geometria de 3 antenas é pobre. Sem palpite.
2. **Subportadoras**: com 114, quais contribuem? A ablação faz diferença
   mensurável, ou fica dentro do intervalo de confiança?
3. **Fase vs amplitude**: a fase sanitizada supera a amplitude pura? No artigo
   Wi-Pose original, o melhor resultado usava **amplitude pura** — pode se repetir.
4. **Queda S1 → S2**: quanto o modelo aprendeu a pessoa em vez do movimento?
5. **P1 vs P2**: as atividades diárias (espaço livre) são mais fáceis que os
   exercícios de reabilitação (posição fixa)? A intuição diz que sim, porque há
   mais deslocamento de corpo inteiro — mas é para medir.

---

## 11. Referências

- Yang, J., Huang, H., Zhou, Y., Chen, X., Xu, Y., Yuan, S., Zou, H., Lu, C.X., Xie, L.
  **"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing."**
  *NeurIPS 2023, Datasets and Benchmarks Track.*
  [Artigo](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3baf7a39d07e9f4f1e258a412df94521-Abstract-Datasets_and_Benchmarks.html) ·
  [arXiv](https://arxiv.org/abs/2305.10345) ·
  [Toolbox oficial](https://github.com/ybhbingo/MMFi_dataset)
- Jiang, W., et al. **"Towards 3D Human Pose Construction Using WiFi."** MobiCom 2020.
- Ren, Y., et al. **"Winect: 3D Human Pose Tracking for Free-form Activity."** IMWUT 2021.
- Análise do dataset anterior: [`analise-inicial.md`](analise-inicial.md)
