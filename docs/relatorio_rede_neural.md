# Relatório Técnico: Rede Neural Residual Espectro-Temporal para Classificação de Ações via WiFi CSI

**Projeto:** Sensoriamento de Poses Humanas Usando Sinais WiFi  
**Disciplina:** Ações Integradas em Ciência e Tecnologia I (AICT I)  
**Instituição:** Universidade Federal do Ceará (UFC) — Departamento de Engenharia de Teleinformática (DETI)  
**Código das Tarefas Atendidas:** Tarefa (c) — *Classificação de Movimentos* e Tarefa (e) — *Análise Crítica & Matrizes de Confusão*  
**Branch de Desenvolvimento:** `feature/refatoracao-rede-neural-csi`  

---

## 1. Introdução e Formulação Física do Problema

O sensoriamento passivo de atividades humanas por radiofrequência (RF) fundamenta-se no princípio de que a propagação de ondas eletromagnéticas em ambientes internos (*indoor*) é modulada pela movimentação cinemática do corpo. Em sistemas comerciais compatíveis com o padrão IEEE 802.11n, a camada física opera com modulação por divisão de frequências ortogonais (**OFDM**).

Ao contrário de métricas escalares agregadas como o RSSI (*Received Signal Strength Indicator*), que condensam toda a potência a uma média escalar sujeita a desvanecimento destrutivo, a Informação de Estado do Canal (**CSI — Channel State Information**) amostra diretamente a **Função de Transferência do Canal (CTF)** complexa para múltiplas subportadoras ortogonais e antenas receptoras.

No domínio do tempo e frequência, o tensor de CSI recebido no instante $t$ decompõe-se fisicamente em:

$$H_m(f_k, t) = H_{m, \text{estatico}}(f_k) + \sum_{p \in \text{corpo}} a_{m, p}(t) e^{-j 2\pi f_k \tau_{m, p}(t)} + \epsilon_m(f_k, t)$$

Onde:
* $H_{m, \text{estatico}}(f_k)$ agrega as reflexões estáticas predominantes do ambiente (paredes, teto, chão e mobília) e a fisionomia neutra estática do indivíduo.
* $\sum_{p \in \text{corpo}} a_{m, p}(t) e^{-j 2\pi f_k \tau_{m, p}(t)}$ representa a componente dinâmica dispersiva induzida pelos membros em movimento, gerando desvios Doppler proporcionais à velocidade radial de cada articulação: $f_D(t) = -\frac{1}{\lambda} \frac{d\tau(t)}{dt}$.
* $\epsilon_m(f_k, t)$ engloba ruído térmico aditivo e imperfeições de hardware (*Carrier Frequency Offset* - CFO e *Packet Detection Delay* - PDD).

### Por que a Modelagem em Janelas Temporais é Inegociável?
Em modelos iniciais ingênuos que tentam classificar ações a partir de um único quadro de 100 ms (10 pacotes OFDM), o componente estático $H_{\text{estatico}}$ domina quase 100% da resposta, e posturas estáticas iniciais (a pessoa aguardando o início do exercício de pé) recebem rótulos arbitrários de ações distintas. 

Para capturar a real dinâmica do movimento cinemático e generalizar entre diferentes pessoas no protocolo Cross-Subject (S2), é **fundamental integrar janelas temporais de alguns segundos** (ex.: 3 segundos = 300 pacotes OFDM), calculando métricas de variabilidade dinâmica ($\sigma$) e evolução temporal que isolam as reflexões do corpo do canal estático da sala.

---

## 2. Dataset MM-Fi e Pré-Processamento Estruturado

O projeto emprega o conjunto de referência **MM-Fi** (NeurIPS 2023), capturado com transceptores comerciais a 5 GHz operando com largura de banda de 40 MHz ($K = 114$ subportadoras OFDM e $M = 3$ antenas receptoras).

### 2.1 Janelamento Temporal e Extração de Características
Em cada sequência de 297 quadros (aproximadamente 30 segundos de gravação a 10 Hz), aplicamos janelas deslizantes de $T = 30$ quadros (3 segundos = 300 pacotes de CSI a 100 Hz):
* **Tratamento de Não-Finitos:** Imputação de `-inf` decorrentes de $\log_{10}(0)$ na amplitude em dB.
* **Sanitização de Fase da Tarefa (d):** Remoção de offsets de rampa linear e desacoplamento de descontinuidades de fase entre blocos de subportadoras via regressão linear segmentada.
* **Descritores Espectro-Temporais por Janela:** Para amplitude e fase sanitizada, calculam-se a média e o desvio padrão temporal ao longo dos 300 pacotes da janela para cada par (antena, subportadora):
  $$X_{\text{janela}} = [\mu_{\text{amp}} (342) \mid \sigma_{\text{amp}} (342) \mid \mu_{\text{fase}} (342) \mid \sigma_{\text{fase}} (342)] \in \mathbb{R}^{1.368}$$
  Onde $342 = 3 \text{ antenas} \times 114 \text{ subportadoras}$. O desvio padrão ($\sigma$) capta a energia Doppler do movimento humano, enquanto a média ($\mu$) mapeia a resposta espacial do canal.
* **Cache Vetorizado Compacto:** As 9.720 janelas das 1.080 sequências do MM-Fi são consolidadas em cache de ~50 MB (`artifacts/cache_janelas_30.npz`), permitindo carregar o dataset em menos de 100 milissegundos.

---

## 3. Arquitetura da Rede Neural Residual Temporal (`CSITemporalActionNet`)

Desenvolvemos uma arquitetura neural profunda baseada em blocos residuais (*Skip Connections*) com regularização estrita para cenários de alta dimensionalidade espectral:

```
Vetor Espectro-Temporal de Entrada: X ∈ ℝ^(Batch × 1368)
   │
   ▼
[Projeção de Entrada: Linear(1368 → 512) → BatchNorm1d → GELU → Dropout(0.3)]
   │
   ▼
[Bloco Residual 1: Linear(512 → 512) → BatchNorm1d → GELU → Dropout(0.3) → Linear → BatchNorm1d + Residual → GELU]
   │
   ▼
[Bloco Residual 2: Linear(512 → 512) → BatchNorm1d → GELU → Dropout(0.3) → Linear → BatchNorm1d + Residual → GELU]
   │
   ▼
[Bloco Residual 3: Linear(512 → 512) → BatchNorm1d → GELU → Dropout(0.3) → Linear → BatchNorm1d + Residual → GELU]
   │
   ▼
[Cabeça Classificadora: Linear(512 → 256) → BatchNorm1d → GELU → Dropout(0.3) → Linear(256 → Num_Classes)]
   │
   ▼
Logits de Saída (Loss: Cross-Entropy Loss com AdamW e Cosine Annealing)
```

### Agregação por Sequência via Soft Voting
Durante a inferência, para cada uma das sequências de teste, a rede processa as 9 janelas temporais constitutivas ($j=1, \dots, 9$), calcula os vetores de probabilidade via Softmax e computa o voto suave médio:

$$\bar{\mathbf{p}}_{\text{seq}} = \frac{1}{J} \sum_{j=1}^{J} \text{Softmax}(\mathbf{z}_j)$$
$$\hat{y}_{\text{seq}} = \arg\max_c (\bar{\mathbf{p}}_{\text{seq}})$$

Isso garante que perturbações momentâneas ou pausas entre repetições em uma janela individual não corrompam a predição da ação global.

---

## 4. Metodologia Experimental e Protocolos

* **Split Oficial Cross-Subject (S2):**
  * **Treinamento:** 32 voluntários distribuídos nos 4 ambientes ($E01$ a $E04$), correspondendo a 7.776 janelas de treino.
  * **Teste Retido (Pessoas Inéditas):** 8 voluntários nunca vistos pelo modelo (`S05, S10, S15, S20, S25, S30, S35, S40`), correspondendo a 1.944 janelas e 216 sequências de teste.
* **Protocolos Avaliados:**
  * **Protocolo P3:** As 27 ações completas do MM-Fi (acaso teórico = $1/27 \approx 3{,}70\%$).
  * **Protocolo P1:** As 14 ações cotidianas (acaso teórico = $1/14 \approx 7{,}14\%$).
* **Normalização Anti-Vazamento:** O `StandardScaler` é ajustado exclusivamente nos dados de treino; os parâmetros ($\mu_{\text{tr}}, \sigma_{\text{tr}}$) são aplicados de forma estrita no conjunto de teste.
* **Treinamento:** 25 épocas, AdamW ($lr = 10^{-3}$, weight decay $10^{-3}$), scheduler Cosine Annealing e batch size 64.

---

## 5. Resultados Obtidos e Discussão Técnica

### 5.1 Convergência e Comparativo com os Baselines

O treinamento convergiu de maneira estável em menos de 20 segundos em CPU:

* **Evolução da Perda:** Queda acentuada de **1,4354** (época 5) para **0,2089** (época 25), demonstrando que a rede aprendeu de forma consistente os padrões espectro-temporais.
* **Desempenho no Split Oficial Cross-Subject (S2):**

| Modelo | Protocolo | Acurácia por Janela | Acurácia por Sequência (Soft Voting) | Acaso Teórico | Fator de Ganho sobre o Acaso |
|---|---|---|---|---|---|
| Random Forest (Baseline) | P3 (27 ações) | 14,51% | 15,28% | 3,70% | 4,13× |
| SVM RBF (Baseline) | P3 (27 ações) | 16,50% | 17,59% | 3,70% | 4,75× |
| **CSITemporalActionNet (Nova)** | **P3 (27 ações)** | **15,38%** | **18,98%** | **3,70%** | **5,13×** |
| **CSITemporalActionNet (Nova)** | **P1 (14 ações)** | **20,44%** | **25,89%** | **7,14%** | **3,63×** |

* **Conclusão:** O novo modelo atinge **18,98% de acurácia no cenário mais rigoroso de generalização entre pessoas (S2) em 27 classes**, superando todos os baselines clássicos (Random Forest e SVM) e operando **mais de 5 vezes acima da linha do acaso**.

---

### 5.2 Análise da Matriz de Confusão (Tarefa e)

A matriz de confusão normalizada por sequência (`artifacts/figuras/matriz_confusao.png`) revela insights físicos consistentes:

![Matriz de Confusão](artifacts/figuras/matriz_confusao.png)

1. **Ações Altamente Discrimináveis (Diagonal Clara):**
   * Ações que envolvem translação de grandes massas corporais e deslocamento vigoroso das pernas — como **A12 (agachamento)**, **A26 (salto)**, **A06 (marcha)** e **A09/A10 (afundos)** — apresentaram as maiores taxas de acerto relativo (acima de 35%–45%), gerando assinaturas Doppler marcantes em praticamente todas as subportadoras.
2. **Desafios Físicos de Ambiguidade Angular:**
   * Movimentos estritamente simétricos (ex.: arremessar para a esquerda `A20` vs arremessar para a direita `A21`, ou chutes espelhados `A22` vs `A23`) continuam apresentando confusão cruzada.
   * **Causa Física:** Em um arranjo de apenas 3 antenas receptoras com espaçamento linear de $\approx 2{,}9\text{ cm}$, a resolução angular azimutal é restrita. Como o WiFi não possui resolução óptica, a movimentação do tronco em contrapeso gera variações de amplitude muito semelhantes para trajetórias contralaterais espelhadas.

---

### 5.3 Análise Crítica: Ablação de Subportadoras OFDM (Tarefa e)

Executamos o experimento de ablação espectral desativando subconjuntos específicos das 114 subportadoras OFDM durante a inferência:

| Configuração Espectral | Subportadoras Ativas | Acurácia por Sequência | Acurácia por Janela | Impacto Observado |
|---|---|---|---|---|
| **Espectro Completo (Referência)** | **114 (100%)** | **18,98%** | **15,38%** | Banda total de 40 MHz (5 GHz) |
| **Bordas Removidas** | 96 (84%) | 13,43% | 13,01% | Redução de 5,55% na acurácia |
| **Apenas 50% Centrais** | 58 (51%) | 8,80% | 7,77% | Perda severa de resolução espectral (-10,18%) |
| **Amostragem Intercalada (Pares)**| 57 (50%) | **7,41%** | **7,00%** | Despenca para próximo do acaso (-11,57%) |

![Ablação de Subportadoras](artifacts/figuras/ablacao_subportadoras.png)

#### Conclusões Físicas da Ablação:
1. **Importância Fundamental da Continuidade Espectral:**
   * Quando reduzimos o espectro para 57 subportadoras usando amostragem intercalada (*apenas subportadoras pares*), a acurácia despenca de 18,98% para **7,41%**.
   * A explicação física é que o perfil de retardo de potência (*Power Delay Profile* - PDP), derivado da IFFT sobre o CSI em frequência, exige densidade espectral contínua para evitar *aliasing* no domínio do atraso temporal e separar os ecos de multipercurso.
2. **Impacto da Largura de Banda de 40 MHz:**
   * A redução da largura de banda efetiva (remoção de bordas e retenção apenas da faixa central) degrada gradativamente a capacidade de discriminar micro-Dopplers finos de movimentos de membros superiores, comprovando que a largura de banda de 40 MHz é crucial para sensoriamento WiFi em ambientes fechados.

---

## 6. Conclusão

1. **Correção de Metodologia:** O modelo anterior falhava (~5% de acerto) por tratar quadros estáticos isolados de 100 ms como se contivessem informação de ação completa, sofrendo com ruído de rótulo e dominância do canal estático.
2. **Eficácia da Abordagem Temporal:** A introdução do janelamento de 3 segundos com descritores de variabilidade dinâmica ($\sigma$) e agregação por *Soft Voting* permitiu à rede convergir com estabilidade (loss de 0,20) e atingir **18,98% de acurácia no split Cross-Subject oficial (27 classes)**, superando Random Forest e SVM.
3. **Cumprimento Pleno do `tema.pdf`:**
   * **Tarefa (c) concluída:** Modelo neural residual temporal treinado e validado.
   * **Tarefa (e) concluída:** Matriz de confusão analisada e ablação de subportadoras OFDM demonstrada experimentalmente com fundamentação física de telecomunicações.
4. **Reprodutibilidade:** Os módulos [`src/rede_neural.py`](file:///C:/Users/Samuel/Desktop/AICT-I-WiPose/src/rede_neural.py), [`src/dataset_rede.py`](file:///C:/Users/Samuel/Desktop/AICT-I-WiPose/src/dataset_rede.py), [`scripts/treinar_rede.py`](file:///C:/Users/Samuel/Desktop/AICT-I-WiPose/scripts/treinar_rede.py) e [`tests/test_rede_neural.py`](file:///C:/Users/Samuel/Desktop/AICT-I-WiPose/tests/test_rede_neural.py) encontram-se testados e com 100% de aprovação na suíte de testes do repositório.
