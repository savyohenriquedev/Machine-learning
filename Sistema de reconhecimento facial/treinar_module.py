import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from keras_facenet import FaceNet
import pickle

# ============================================
# Inicializando o modelo FaceNet (gera vetores)
# ============================================
embedder = FaceNet()

# Diretório onde estão as imagens organizadas por pessoa
pasta_raiz = 'imagens'

# Onde vamos guardar os embeddings e os rótulos (nomes)
embeddings = []
rotulos = []

# Onde vamos guardar os metadados (idade e cargo)
metadata = {}

# Percorro cada pasta (pessoa) dentro da pasta raiz
for nome_pasta in os.listdir(pasta_raiz):
    caminho_pessoa = os.path.join(pasta_raiz, nome_pasta)
    if not os.path.isdir(caminho_pessoa):
        continue  # pula se não for pasta

    # Extrai nome, idade e cargo a partir do nome da pasta
    try:
        partes = nome_pasta.split('_')
        nome = partes[0] + ' ' + partes[1]
        idade = int(partes[2])
        cargo = '_'.join(partes[3:])
    except Exception as e:
        print(f"Erro no nome da pasta: {nome_pasta}")
        continue

    # Salvo os dados da pessoa
    metadata[nome] = {'idade': idade, 'cargo': cargo}

    # Leio todas as imagens dessa pessoa
    for arquivo in os.listdir(caminho_pessoa):
        caminho_imagem = os.path.join(caminho_pessoa, arquivo)
        img = cv2.imread(caminho_imagem)

        if img is None:
            continue  # pula imagens corrompidas

        # Redimensiono e normalizo a imagem
        img = cv2.resize(img, (160, 160))
        img = img.astype('float32') / 255.0

        # Extraio o vetor de características (embedding)
        embedding = embedder.embeddings([img])[0]

        embeddings.append(embedding)
        rotulos.append(nome)

# ============================
# Treinamento do classificador
# ============================
print("Treinando classificador KNN...")
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(embeddings, rotulos)

# ============================
# Salvando os arquivos finais
# ============================

# Modelo KNN
with open('knn_classifier.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Metadados (nome → idade, cargo)
with open('pessoas_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Modelo treinado e salvo com sucesso.")
