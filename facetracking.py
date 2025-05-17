# Feito por: Savyo Rocha
# Importando as bibliotecas necessárias para o sistema funcionar
import cv2  # para capturar vídeo em tempo real da webcam
from mtcnn import MTCNN  # detector de rostos baseado em deep learning
from keras_facenet import FaceNet  # modelo para gerar embeddings faciais
import numpy as np  # operações numéricas
from sklearn.neighbors import KNeighborsClassifier  # classificador baseado em similaridade
import pickle  # para carregar o modelo treinado previamente

# ===============================
# Inicialização dos componentes
# ===============================

# Instancia o detector de rostos MTCNN
detector = MTCNN()

# Carrega o modelo FaceNet que transforma o rosto em um vetor (embedding)
embedder = FaceNet()

# Aqui estou carregando um classificador KNN treinado anteriormente
# Esse classificador vai pegar os embeddings e dizer de quem é o rosto
with open('knn_classifier.pkl', 'rb') as f:
    knn = pickle.load(f)

# Também carrego o dicionário com os metadados das pessoas reconhecidas
# Exemplo de formato: { 'Savyo Rocha': {'idade': 13, 'cargo': 'Cientista de dados'}, ... }
with open('pessoas_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)


# ===================================
# Função para processar um rosto só
# ===================================
def processar_rosto(face_img):
    # Redimensiono a imagem do rosto para o formato esperado pelo modelo (160x160)
    face_resized = cv2.resize(face_img, (160, 160))

    # Normalizo os pixels (o modelo espera valores entre 0 e 1)
    face_normalized = face_resized.astype('float32') / 255.0

    # Extraio o embedding com o modelo FaceNet
    embedding = embedder.embeddings([face_normalized])[0]

    return embedding


# ======================================
# Início da captura de vídeo em tempo real
# ======================================
cap = cv2.VideoCapture(0)  # 0 representa a webcam padrão

while True:
    # Lê o frame atual da webcam
    ret, frame = cap.read()
    if not ret:
        break  # se não conseguir capturar o vídeo, sai do loop

    # Faz a detecção de rostos na imagem capturada
    faces = detector.detect_faces(frame)

    # Percorre todos os rostos detectados
    for face in faces:
        # Extraio as coordenadas do rosto (x, y, largura, altura)
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)  # evito valores negativos

        # Recorto o rosto da imagem original
        face_img = frame[y:y + h, x:x + w]

        # Extraio o vetor de características (embedding) do rosto
        embedding = processar_rosto(face_img)

        # Faz a previsão com o classificador (o nome da pessoa)
        nome_previsto = knn.predict([embedding])[0]

        # Recupero os dados da pessoa no dicionário de metadados
        dados = metadata.get(nome_previsto, {'idade': '?', 'cargo': '?'})
        idade = dados['idade']
        cargo = dados['cargo']
        # Desenha o retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Fonte utilizada: padrão, clara e legível
        fonte = cv2.QT_FONT_NORMAL
        tamanho_fonte = 0.5
        espessura = 1
        cor = (0, 255, 0)  # verde

        # Linha inicial para o texto (logo abaixo do retângulo)
        linha_base = y + h + 20
        espaco_entre_linhas = 25  # espaço entre as linhas

        # Exibe os dados da pessoa, um abaixo do outro
        cv2.putText(frame, f"Nome: {nome_previsto}", (x, linha_base), fonte, tamanho_fonte, cor, espessura)
        cv2.putText(frame, f"Idade: {idade} anos", (x, linha_base + espaco_entre_linhas), fonte, tamanho_fonte, cor,
                    espessura)
        cv2.putText(frame, f"Cargo: {cargo}", (x, linha_base + 2 * espaco_entre_linhas), fonte, tamanho_fonte, cor,
                    espessura)

    # Mostra o frame com os rostos identificados
    cv2.imshow("Reconhecimento Facial em Tempo Real", frame)

    # Se a tecla 'q' for pressionada, o loop para
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libero a câmera e fecho as janelas
cap.release()
cv2.destroyAllWindows()
