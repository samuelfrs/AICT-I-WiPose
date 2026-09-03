# Diretório do Dataset Wi-Pose

Coloque os arquivos .mat do dataset Wi-Pose dentro deste diretório (dataset/).

### Download do Dataset Original:
- **Repositório oficial:** [NjtechCVLab/Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset)
- **Google Drive:** [Download via Google Drive](https://drive.google.com/file/d/1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9/view?usp=sharing)
- **Baidu Netdisk:** [Download via Baidu Netdisk](https://pan.baidu.com/s/14TbkIRPQN1DktDir9N2Y3A) (Senha: wqeq)

### Estrutura dos Arquivos (.mat):
Cada pacote contém:
- csi_serial: Informação de Estado do Canal (5 × 30 × 3 × 3), representando 5 amostras temporais de CSI por anotação visual, 30 subportadoras OFDM e 3x3 antenas transmissoras e receptoras.
- jointsVector: Coordenadas das 18 articulações do esqueleto no formato (x, y, c, c), onde (x, y) é a posição e c é a confiança gerada pelo AlphaPose.
- jointsMatrix: Matriz de distâncias/deslocamentos relativos entre articulações.
