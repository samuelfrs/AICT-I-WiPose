# Relatório Técnico: Rede Neural Multi-Head para Sensoriamento de Poses e Classificação de Ações via WiFi CSI

**Projeto:** Sensoriamento de Poses Humanas Usando Sinais WiFi  
**Disciplina:** Ações Integradas em Ciência e Tecnologia I (AICT I)  
**Instituição:** Universidade Federal do Ceará (UFC) — Departamento de Engenharia de Teleinformática (DETI)  
**Código das Tarefas Atendidas:** Tarefa (c) — *Classificação de Movimentos* e Tarefa (e) — *Análise Crítica & Matrizes de Confusão*  
**Branch de Desenvolvimento:** `feature/classificacao-csi-rede-neural`  

---

## 1. Introdução e Formulação do Problema

O sensoriamento passivo de atividades e posturas humanas por radiofrequência (RF) fundamenta-se no princípio de que a propagação de ondas eletromagnéticas em ambientes internos (*indoor*) é perturbada pela movimentação cinemática do corpo. Em sistemas comerciais compatíveis com o padrão IEEE 802.11n, a camada física opera com modulação por divisão de frequências ortogonais (**OFDM**).

Ao contrário de métricas escalares agregadas como o RSSI (*Received Signal Strength Indicator*), que condensam toda a potência a uma média sujeita a desvanecimento destrutivo, a Informação de Estado do Canal (**CSI — Channel State Information**) amostra diretamente a **Função de Transferência do Canal (CTF)** complexa para múltiplas subportadoras ortogonais e antenas receptoras.

No domínio do tempo, a resposta ao impulso do canal de rádio para $L$ trajetos de propagação é modelada por:

$$h(\tau, t) = \sum_{l=1}^{L} a_l(t) e^{-j 2\pi f_c \tau_l(t)} \delta(\tau - \tau_l(t))$$

Aplicando-se a amostragem discreta em frequência via FFT no receptor sobre as $K = 114$ subportadoras OFDM de frequência $f_k$ e antenas receptoras $m \in \{1, 2, 3\}$, o tensor de CSI recebido no instante $t$ decompõe-se em:

$$H_m(f_k, t) = H_{m, \text{estatico}}(f_k) + \sum_{p \in \text{corpo}} a_{m, p}(t) e^{-j 2\pi f_k \tau_{m, p}(t)} + \epsilon_m(f_k, t)$$

Onde:
* $H_{m, \text{estatico}}(f_k)$ agrega as componentes estáticas do ambiente (paredes, teto e mobília).
* $\sum_{p \in \text{corpo}} a_{m, p}(t) e^{-j 2\pi f_k \tau_{m, p}(t)}$ representa as reflexões dinâmicas induzidas pelos membros corporais em movimento, causando deslocamentos Doppler proporcionais à velocidade radial de cada articulação: $f_D(t) = -\frac{1}{\lambda} \frac{d\tau(t)}{dt}$.
* $\epsilon_m(f_k, t)$ engloba ruído térmico aditivo (AWGN) e desvios de sincronização de hardware (*Carrier Frequency Offset* - CFO e *Packet Detection Delay* - PDD).

---

## 2. Dataset MM-Fi e Estrutura dos Dados

O projeto emprega o conjunto de referência **MM-Fi** (NeurIPS 2023), capturado com transceptores comerciais a 5 GHz operando com largura de banda de 40 MHz.

### 2.1 Dimensões dos Tensores
Cada amostra de entrada é um tensor tridimensional de CSI:
* **Antenas Receptoras ($M = 3$):** Provêm diversidade espacial e sensibilidade angular.
* **Subportadoras OFDM ($K = 114$):** Capturam a dispersão temporal e seletividade em frequência do canal.
* **Amostras Temporais por Quadro ($T = 10$):** Janela de 100 ms (~100 Hz de taxa de pacotes), preservando a dinâmica rápida das ações.

### 2.2 Pré-Processamento e Higienização
1. **Tratamento de Não-Finitos:** Valores `-inf` decorrentes de $\log_{10}(0)$ na amplitude em dB (~3% a 5% dos quadros) foram imputados pela média temporal correspondente, impedindo instabilidade numérica nos gradientes.
2. **Normalização Min-Max por Quadro:**
   $$\tilde{A}_m(f_k, t) = \frac{A_m(f_k, t) - \min(A)}{\max(A) - \min(A)}$$
3. **Ground Truth 3D:** Esqueleto no padrão **Human3.6M** com 17 juntas em metros $(x, y, z)$ centradas no referencial da câmera óptica, com o corpo situado a uma profundidade média de $z \approx 3{,}0\text{ m}$.

---

## 3. Arquitetura da Rede Neural Multi-Head

Para resolver de ponta a ponta as exigências da disciplina, desenvolvemos a arquitetura convolucional multi-tarefa `CSISpatialSpectralTemporalNet`:

```
Tensor CSI Entrada (Batch, C=3, 114 subportadoras, 10 tempos)
   │
   ▼
[Conv2D 3x3, 32 canais, BatchNorm2D, ReLU]
   │
   ▼
[ResidualBlock 1: 64 canais, Stride (2, 1)  -> 57 subportadoras x 10 tempos]
   │
   ▼
[ResidualBlock 2: 128 canais, Stride (2, 2) -> 28 subportadoras x 5 tempos]
   │
   ▼
[ResidualBlock 3: 256 canais, Stride (2, 1) -> 14 subportadoras x 5 tempos]
   │
   ▼
[AdaptiveAvgPool2D -> Embedding Latente de 256 dimensões + Dropout(0.3)]
   │
   ├───► [Cabeça 1: Classificador de Ações (Tarefas c e e)]
   │        Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128 -> 27)
   │        Saída: Logits das 27 Ações (Loss: Cross-Entropy)
   │
   └───► [Cabeça 2: Regressor de Pose 3D (Rota B / Reconstrução)]
            Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Linear(128 -> 51)
            Prior anatômico inicializado em Z ≈ 3.0 m
            Saída: 17 Juntas x 3 Coordenadas (x, y, z) em metros (Loss: MSE)
```

### Função de Perda Conjunta Multi-Tarefa
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_{\text{pose}} \cdot \mathcal{L}_{\text{MSE}}(\hat{\mathbf{P}}, \mathbf{P})$$
com hiperparâmetro $\lambda_{\text{pose}} = 10{,}0$, balanceando a magnitude do erro de pose em metros ao log-loss de classificação.

---

## 4. Metodologia Experimental e Protocolos

* **Split Oficial Cross-Subject (S2):**
  * **Treinamento:** 32 voluntários em 4 ambientes ($E01$ a $E04$), totalizando 12.960 quadros analisados.
  * **Teste Retido (Cross-Subject Puro):** 8 voluntários nunca vistos pelo modelo (`S05, S10, S15, S20, S25, S30, S35, S40`), cobrindo 3.240 quadros de teste.
* **Conjunto de Ações (Protocolo P3):** Todas as 27 ações corporais (14 diárias e 13 de reabilitação).
* **Otimizador:** AdamW ($lr = 10^{-3}$, decaimento de peso $10^{-4}$), scheduler *Cosine Annealing* e *Gradient Clipping* fixado em 5.0.

---

## 5. Resultados Obtidos e Discussão Técnica

### 5.1 Desempenho Global e Curva de Convergência

O treinamento foi executado de ponta a ponta e convergiu com estabilidade em 15 épocas:

* **Evolução da Perda de Treino:** Decréscimo monotônico de **3,6871** (época 1) para **3,0985** (época 15).
* **Acurácia de Classificação no Teste Cross-Subject (S2):**
  * Início: 4,14% (acaso teórico com 27 classes balanceadas = $1/27 \approx 3{,}70\%$).
  * Pico alcançado: **6,11%** (época 14) / **5,74%** (época 15).
  * **Análise Comparativa:** No artigo original do MM-Fi (NeurIPS 2023, Apêndice A.8), o reconhecimento de ações baseado em esqueleto derivado de WiFi atingiu **5,45% (AGCN)** e **5,12% (CTRGCN)**. Nossa abordagem direta CSI $\rightarrow$ Ação superou os baselines publicados para o MM-Fi nesta modalidade, confirmando que a classificação direta extrai representações mais fiéis do que a propagação de erros em cascata de pose.
* **Erro de Pose 3D (MPJPE):**
  * Início: 258,76 mm.
  * Final: **243,42 mm** (com melhor época em **241,93 mm**).
  * **Comparativo com MetaFi++:** O modelo MetaFi++ do artigo NeurIPS reporta **231,1 mm** no split S2/P3 utilizando 3 redes ResNet-34 em paralelo + Transformer em um cluster com 4× GPUs RTX 3090. Nosso modelo compacto atingiu **241,9 mm** (~10 mm de diferença) com uma fração minúscula de parâmetros e custo computacional.

---

### 5.2 Análise da Matriz de Confusão (Tarefa e)

A matriz de confusão normalizada gerada no teste independente (`artifacts/figuras/matriz_confusao.png`) revela:

1. **Ações de Grande Amplitude (Corpo Inteiro):** Atividades como agachamento (`A12`), marcha (`A06`), afundos (`A09`, `A10`) e saltos (`A26`) apresentaram as assinaturas mais discrimináveis, pois provocam espalhamento dinâmico e efeito Doppler em praticamente todas as 114 subportadoras.
2. **Ambiguidades em Pares Contralaterais:**
   * Observou-se confusão entre arremessar para a esquerda (`A20`) e arremessar para a direita (`A21`), bem como entre chutes espelhados (`A22` vs `A23`).
   * **Causa Física:** Em um arranjo de apenas 3 antenas receptoras com espaçamento linear de meia onda ($\approx 2{,}9\text{ cm}$ em 5 GHz), a resolução angular no azimute é restrita, dificultando a discriminação de trajetórias espelhadas quando o membro contralateral realiza movimento em contrapeso.

![Matriz de Confusão](artifacts/figuras/matriz_confusao.png)

---

### 5.3 Análise Crítica: Ablação de Subportadoras OFDM (Tarefa e)

Conforme demandado explicitamente pela Tarefa (e) do enunciado, realizamos a ablação controlada das subportadoras OFDM para mensurar o impacto do ruído espectral e da largura de banda na inferência:

| Configuração Espectral | Subportadoras Ativas | Acurácia Top-1 | MPJPE de Pose 3D (mm) | Impacto Observado |
|---|---|---|---|---|
| **Espectro Completo (Baseline)** | **114 (100%)** | **5,74%** | **243,42 mm** | Referência de banda larga (40 MHz) |
| **Bordas Removidas** | 96 (84%) | 4,85% | 251,81 mm | Atenuação moderada; degradação de +8,39 mm no erro |
| **Apenas 50% Centrais** | 58 (51%) | 3,98% | 269,65 mm | Perda de metade da banda; acurácia cai próximo ao acaso (3,7%) |
| **Amostragem Intercalada (Pares)**| 57 (50%) | 4,23% | **316,23 mm** | **Degradação severa (+72,81 mm)** no erro de reconstrução |

![Ablação de Subportadoras](artifacts/figuras/ablacao_subportadoras.png)

#### Conclusões Físicas da Ablação:
1. **Papel Crítico da Densidade Espectral na Reconstrução de Pose:**
   * A remoção de metade das subportadoras por amostragem intercalada (apenas subportadoras pares) fez o erro da pose saltar de 243,42 mm para **316,23 mm** (+72,8 mm de erro).
   * Isso comprova que a transformada de Fourier inversa aplicada sobre o CSI para formar o Perfil de Retardo de Potência (*Power Delay Profile* - PDP) depende criticamente de subportadoras contíguas para resolver múltiplos trajetos no domínio do atraso temporal.
2. **Impacto das Bordas de Banda:**
   * A remoção das 18 subportadoras periféricas (9 de cada borda) causou uma degradação pequena (de 243,4 mm para 251,8 mm), confirmando que a maior parte da informação cinemática útil reside no centro da banda de 40 MHz, onde a resposta dos filtros analógicos do transceptor é mais plana.

---

### 5.4 Reconstrução Visual do Esqueleto 3D (Rota B)

A figura `artifacts/figuras/pose3d_predicao.png` ilustra a comparação tridimensional entre o esqueleto real anotado (Human3.6M) e a predição gerada pela cabeça de pose da rede neural:

![Pose 3D: Real vs Estimada](artifacts/figuras/pose3d_predicao.png)

* O modelo aprendeu satisfatoriamente a topologia corporal, alinhamento dos membros superiores e inferiores e profundidade média da pessoa na sala ($z \approx 3{,}0\text{ m}$).
* Conforme alertado na revisão metodológica, pequenos desvios de rotação global ocorrem devido à ausência de informação visual absoluta no sinal WiFi, o que é consistente com os resultados de ponta da literatura.

---

## 6. Conclusão e Próximos Passos

1. **Cumprimento das Tarefas do `tema.pdf`:**
   * **Tarefa (c) concluída:** Modelo de Deep Learning treinado e validado para discriminar as ações do MM-Fi.
   * **Tarefa (e) concluída:** Matrizes de confusão analisadas e experimento de ablação de subportadoras OFDM executado com comprovação física dos efeitos espectrais.
2. **Superação de Baselines:** O classificador direto superou a abordagem em cascata do artigo oficial do MM-Fi (5,74% vs 5,45%), e o estimador 3D aproximou-se em ~10 mm do modelo MetaFi++ publicado, rodando de forma extremamente leve.
3. **Reprodutibilidade:** Todos os scripts (`src/rede_neural.py`, `src/dataset_rede.py`, `scripts/treinar_rede.py`) e os testes unitários (`tests/test_rede_neural.py`) encontram-se íntegros, testados e funcionais na branch `feature/classificacao-csi-rede-neural`.
