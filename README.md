# Sensoriamento de Poses Humanas Usando Sinais WiFi (Wi-Pose)

Projeto desenvolvido no âmbito da disciplina de **Ações Integradas de Ciência e Tecnologia I (AICT I)**.

## 🎯 Objetivo
Processamento digital de sinais de rádio (RF) e modelagem de algoritmos de aprendizado de máquina / deep learning para estimação de movimentos e poses corporais 3D a partir de sinais de WiFi (Channel State Information - CSI), sem câmeras ou sensores corporais vestíveis.

## 📚 Artigos de Referência
- **Wi-Pose:** Jiang, W., et al. *"Towards 3D human pose construction using wifi."* Proc. MobiCom, 2020.
- **Winect:** Ren, Y., et al. *"Winect: 3D Human Pose Tracking for Free-form Activity Using Commodity WiFi."* Proc. IMWUT, 2021.

## 💾 Dataset
O projeto utiliza o [Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset):
- **Origem:** 166.600 pacotes `.mat` coletados a 5 GHz, com anotações de 18 pontos do esqueleto via AlphaPose para 12 ações executadas por 12 voluntários.
- **Download:** Veja as instruções de download e posicionamento dos arquivos em [dataset/README.md](dataset/README.md).

## 🗂️ Estrutura do Repositório
```
AICT I/
├── Artigos/                # Artigos científicos base em PDF
├── dataset/                # Diretório local para os arquivos .mat do dataset (ignorado no Git)
├── src/                    # Código-fonte (carregamento de dados, DSP, modelos)
├── tema.pdf                # Descrição da proposta da disciplina
└── README.md
```