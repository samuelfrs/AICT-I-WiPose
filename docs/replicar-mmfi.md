# Replicar o baseline do MM-Fi — o que existe pronto e o que falta

**Código da tarefa:** `RB-03`
**Data:** 10/09/2026

Leitura do artigo completo (versão v2, 20 páginas com apêndice), do toolbox
oficial e do código do MetaFi++. Este documento responde a uma pergunta só:
**o que significa, na prática, "replicar o que eles fizeram"?**

---

## 1. ⚠️ Antes de tudo: replicar o MM-Fi é *estimação de pose*, não classificação

Isso precisa estar claro antes de escrever uma linha de código.

**Os baselines do MM-Fi são todos de estimação de pose 3D.** As tabelas do artigo
(MPJPE / PA-MPJPE) medem quão perto o esqueleto previsto fica do anotado. Não há
nenhum baseline de classificação de ação a partir do CSI.

Ou seja, replicar o trabalho deles = **Rota B** da análise anterior, não a Rota A.

| | Replicar o MM-Fi | A tarefa (c) do `tema.pdf` |
| --- | --- | --- |
| Saída | 17 juntas × (x, y, z) | 1 rótulo entre 27 |
| Métrica | MPJPE, PA-MPJPE (mm) | Acurácia, matriz de confusão |
| Tem baseline publicado? | ✅ Sim, 9 configurações | ❌ Não |
| É o que o enunciado pede? | ⚠️ Parcialmente (é "regressor") | ✅ Sim ("classificar") |

O `tema.pdf` diz *"projetar um classificador **ou regressor** simplificado"* — então
regressão de pose **cabe** no enunciado. Mas a tarefa (c) fala explicitamente em
"classificar e discriminar ações", e a tarefa (e) pede "matrizes de confusão", que
só existem em classificação.

**As duas coisas não se excluem.** Um caminho razoável é replicar o baseline de
pose (dá números comparáveis com a literatura, que é ótimo para o relatório) e
depois fazer a classificação, que é o que fecha as tarefas (c) e (e).

---

## 2. O que já está pronto — e é bastante

O toolbox oficial ([ybhbingo/MMFi_dataset](https://github.com/ybhbingo/MMFi_dataset))
entrega o andaime inteiro:

| Arquivo | O que dá |
| --- | --- |
| `mmfi_lib/mmfi.py` | Dataloader PyTorch completo, todas as modalidades |
| `config.yaml` | Os 3 splits e os 3 protocolos, prontos |
| `mmfi_lib/evaluate.py` | MPJPE e PA-MPJPE, com Procrustes implementado |
| `example.py` | Esqueleto do laço de treino, com os `TODO` marcados |

Ou seja: **carregamento, divisão treino/teste e métricas não precisam ser
escritos.** Isso elimina a maior parte do trabalho chato e do risco de erro
metodológico.

### 2.1 Como o loader oficial trata o CSI

Vale ler com atenção, porque tem três decisões embutidas:

```python
data = scio.loadmat(frame)['CSIamp']         # 1. SÓ AMPLITUDE
data[np.isinf(data)] = np.nan                # 2. trata os -inf
for i in range(10):
    temp_col = data[:, :, i]
    if np.count_nonzero(temp_col != temp_col) != 0:
        temp_col[np.isnan(temp_col)] = temp_col[temp_col == temp_col].mean()
data = (data - np.min(data)) / (np.max(data) - np.min(data))   # 3. min-max POR QUADRO
```

1. 🚨 **`CSIphase` nunca é lida.** Verifiquei: a string "phase" **não aparece em
   lugar nenhum** do toolbox. O baseline oficial de WiFi joga a fase fora.
2. Confirma o problema dos `-inf` que encontramos — e mostra a solução deles:
   virar `NaN` e imputar pela média da fatia temporal.
3. Normalização **min-max por quadro**. Isso é conveniente (não precisa de
   estatística de treino, então não vaza), mas **destrói a amplitude absoluta** —
   só a forma relativa do espectro sobrevive.

**Isto é uma oportunidade.** Se o baseline oficial usa só amplitude, então usar a
fase sanitizada (tarefa d) é *ir além do baseline*, com um número de referência
para comparar. É o tipo de coisa que rende uma seção boa no relatório.

### 2.2 Detalhes do split que corrigem a análise anterior

O `config.yaml` mostra o split cross-subject oficial:

```
teste: S05, S10, S15, S20, S25, S30, S35, S40
```

São **2 sujeitos de cada ambiente** — então o S2 é um teste cross-subject limpo,
com os 4 ambientes presentes nos dois lados. Corrige a ressalva que eu tinha feito
sobre S2 e S3 estarem emaranhados: o split oficial já cuida disso.

*(Duas inconsistências menores: o `config.yaml` usa `ratio: 0.8` para o S1,
enquanto o artigo diz 3:1 (0,75); e o apêndice A.8 usa 30/10 sujeitos, contra os
32/8 do texto principal.)*

---

## 3. O baseline de WiFi: MetaFi++ / WPFormer

**Artigo:** Zhou, Y., et al. *"MetaFi++: WiFi-Enabled Transformer-Based Human Pose
Estimation for Metaverse Avatar Simulation."* IEEE IoT Journal, 10(16), 2023.
**Código:** [pridy999/metafi_pose_estimation](https://github.com/pridy999/metafi_pose_estimation)

### Arquitetura (lida do código)

```
entrada (B, 2, 3, 114, T)
   │
   ├── separa as 3 antenas ──> para cada uma:
   │        transpõe e achata  ──> redimensiona para [136, 32]
   │        ResNet-34 pré-treinada (layers 1 a 4)   ──> (B, 512, h, w)
   │
   ├── concatena as 3 saídas no eixo da largura
   ├── BatchNorm2d(512)
   ├── ChannelTransformer  (self-attention ENTRE as antenas)
   ├── decode: Conv 512→32 → Conv 32→2
   ├── AvgPool
   └── saída (B, 17, 2)
```

A ideia central é a atenção **entre antenas** — o transformer aprende quanto pesar
cada uma das 3 antenas, em vez de tratá-las como canais fixos.

### Hiperparâmetros (do `train.py` original)

| | |
| --- | --- |
| Otimizador | SGD, lr = 0,001, momentum = 0,9 |
| Perda | MSE (ponderada por confiança, no original 2D) |
| Batch | 32 |
| Épocas | 50, com decaimento linear a partir da 20ª |

### ⚠️ Duas adaptações obrigatórias

O código público é do MetaFi++ original, que **não é o do MM-Fi**:

1. **Saída 2D → 3D.** O decoder termina em `Conv2d(32, 2, ...)` e `BatchNorm1d(2)`
   — duas coordenadas. Para o MM-Fi precisa virar 3.
2. **Eixo temporal 32 → 10.** O código espera 32 amostras temporais; o MM-Fi tem
   10. (O próprio loader do MM-Fi tem o comentário revelador `for i in range(10):  # 32`.)

O artigo do MM-Fi diz que usou "os hiperparâmetros dos artigos originais", mas
**não publicou o código adaptado** — a frase *"todos os códigos ... podem ser
consultados no link"* aponta para o toolbox, que não tem os modelos. Então essa
adaptação é nossa.

---

## 4. E o mmWave ("rádio")?

Bem mais pesado. Não recomendo como ponto de partida:

| | WiFi | mmWave |
| --- | --- | --- |
| Formato | `.mat`, tensor fixo `(3, 114, 10)` | `.bin`, nuvem de pontos `(N, 5)` **de tamanho variável** |
| Baseline | MetaFi++, código público | **Point Transformer**, código não adaptado |
| Pré-processamento | min-max, pronto no toolbox | Agregação de quadros consecutivos, *padding* de nuvem |
| Resultado (S1/P3) | 197,1 mm MPJPE | **117,0 mm** — melhor |

O mmWave dá resultado melhor, mas exige lidar com nuvens de pontos de tamanho
variável e uma arquitetura bem mais complexa. **Sugestão: WiFi primeiro.** Se
sobrar tempo, mmWave vira comparação — e aí "WiFi vs mmWave no mesmo dataset" é
uma seção forte de análise crítica.

E tem um argumento de escopo: o `tema.pdf` é sobre **sinais WiFi**. O mmWave é
bônus, não obrigação.

---

## 5. O alvo: os números a bater

Do artigo, MPJPE / PA-MPJPE em mm, WiFi com MetaFi++:

| Split | P1 (14 diárias) | P2 (13 reabilitação) | P3 (todas as 27) |
| --- | --- | --- | --- |
| **S1** aleatório | 186,9 / 120,7 | 213,5 / 121,4 | **197,1 / 121,2** |
| **S2** sujeito novo | 222,3 / 125,4 | 247,0 / 122,7 | 231,1 / 124,0 |
| **S3** ambiente novo | 367,8 / 121,0 | 360,2 / 117,2 | 369,5 / 116,0 |

**Alvo mínimo de uma replicação bem-sucedida: chegar perto de 197 mm em S1/P3.**
Se der entre ~180 e ~220, está replicado. Um número muito melhor é motivo de
desconfiança, não de comemoração — provavelmente é vazamento.

⚠️ Um detalhe de implementação que não bate: o artigo diz que o MPJPE é calculado
*"after aligning the pelvis"*, mas o `evaluate.py` oficial **não alinha nada** —
calcula a distância euclidiana direta. Ou o modelo já emite pose centrada na
pelve, ou a descrição do artigo não corresponde ao código. **Confirmar antes de
comparar números**, senão a comparação não vale.

---

## 6. Esforço estimado e o que teríamos que escrever

| Etapa | Situação |
| --- | --- |
| Carregar dados | ✅ Pronto (toolbox) |
| Splits e protocolos | ✅ Pronto (`config.yaml`) |
| Métricas | ✅ Pronto (`evaluate.py`) |
| **Cache dos dados** | ❌ Nosso. 320.760 `.mat`; ler um a um a cada época é inviável |
| **Modelo** | 🟡 Adaptar o MetaFi++ (2D→3D, 32→10 amostras) |
| **Laço de treino** | ❌ Nosso (o `example.py` só tem os `TODO`) |
| **Experimentos** | ❌ Nossos |

**Cache:** 320.760 × 3 × 114 × 10 em `float32` ≈ **4,4 GB**. Cabe num `.npy`
mapeado em memória. Uma leitura só dos 16 GB de `.mat`, depois nunca mais.

### ⚠️ Realidade de hardware

O artigo rodou em **4× RTX 3090** (24 GB cada). Nós temos **1× RTX 5060, 8 GB**.

O modelo instancia **três ResNet-34** (uma por antena) mais um transformer. Isso
provavelmente cabe em 8 GB com batch 32, mas vai ser lento. Estimativa grosseira:
o treino do S1 (≈256 mil quadros, 50 épocas) pode levar **muitas horas**.

**Mitigações, em ordem de preferência:**
1. Começar pelo **P2** (13 ações, ~520 sequências) em vez do P3 — menos da metade
   do dado, e ainda tem baseline publicado para comparar.
2. Subamostrar quadros por sequência (usar 1 a cada N) para os primeiros testes.
3. Reduzir épocas no ciclo de desenvolvimento e só rodar as 50 no experimento final.
4. Compartilhar os pesos da ResNet entre as três antenas em vez de instanciar três
   — **atenção: isso deixa de ser replicação fiel** e precisa ser declarado.

---

## 7. Plano sugerido

**Fase 0 — provar o caminho (antes de qualquer treino)**
1. Rodar o toolbox oficial na nossa máquina com `modality: wifi-csi` e confirmar
   que o dataloader funciona com o dataset como está em `~/Downloads/datas/`.
2. Construir o cache `.npy` e conferir que o conteúdo bate com a leitura direta.
3. Treinar o modelo por 2 épocas num subconjunto minúsculo, só para fechar o
   caminho feliz do código.

**Fase 1 — replicação**
4. Adaptar o MetaFi++ (saída 3D, 10 amostras temporais).
5. Treinar em **P2 + S1** e comparar com os 213,5 mm publicados.
6. Se bater, repetir em P3 + S1 (alvo: 197,1) e P3 + S2 (alvo: 231,1).

**Fase 2 — ir além do baseline (é aqui que está a contribuição do trabalho)**
7. **Adicionar a fase.** O baseline oficial usa só amplitude. Sanitizar a fase
   (tarefa d) e medir se melhora o MPJPE. Resultado negativo também é entrega.
8. **Classificação das 27 ações** (tarefas c e e), que é o que o enunciado pede e
   não tem baseline publicado.

---

## 8. Decisões que o grupo precisa tomar

1. **Replicar pose 3D, fazer classificação, ou os dois?** Replicar dá números
   comparáveis; classificar é o que a tarefa (c) pede. Minha sugestão: replicar
   primeiro (valida o pipeline inteiro contra um alvo conhecido) e depois
   classificar.
2. **Qual protocolo começar** — sugestão P2, que é o menor.
3. **WiFi só, ou WiFi + mmWave?** Sugestão: WiFi primeiro, mmWave só se sobrar tempo.
4. **Aceitamos desviar da replicação fiel** (ex.: ResNet compartilhada) se o
   hardware apertar? Se sim, tem que estar declarado no relatório.

---

## 9. Referências

- **MM-Fi** (NeurIPS 2023): [arXiv:2305.10345](https://arxiv.org/abs/2305.10345) ·
  [Toolbox](https://github.com/ybhbingo/MMFi_dataset) ·
  [Página do projeto](https://ntu-aiot-lab.github.io/mm-fi)
- **MetaFi++** (IEEE IoT Journal, 2023): [IEEE Xplore](https://ieeexplore.ieee.org/document/10086600/) ·
  [Código](https://github.com/pridy999/metafi_pose_estimation)
- Análise do dataset: [`analise-mmfi.md`](analise-mmfi.md)
