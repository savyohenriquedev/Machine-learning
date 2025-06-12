import pandas as pd
import string
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ====== Funções utilitárias ======

def shannon_entropy(s):
    """
    Calcula a entropia de Shannon da string s.
    Quanto maior a entropia, mais aleatória a senha parece.
    """
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return -sum([p * math.log2(p) for p in prob])

def extract_features(pw):
    """
    Extrai características (features) de uma senha.
    Essas features são usadas como entrada para o modelo de IA.
    """
    return [
        len(pw),  # Tamanho total da senha
        sum(c.isdigit() for c in pw),  # Quantidade de dígitos
        sum(c.isupper() for c in pw),  # Letras maiúsculas
        sum(c.islower() for c in pw),  # Letras minúsculas
        sum(c in string.punctuation for c in pw),  # Símbolos especiais
        shannon_entropy(pw)  # Entropia da senha
    ]

# ====== Base de dados simples para treinamento ======

# Lista com senhas e seus respectivos rótulos (human ou machine)
data = {
    "password": [
        "123qwe", "password123", "letmein2023", "admin2020", "qwerty",
        "Th7$g9L#2m", "8d9F3s!q9T", "@2Fv!s8X", "A1b2C3d4E5", "P$9zKx#1Vu"
    ],
    "label": [
        "human", "human", "human", "human", "human",
        "machine", "machine", "machine", "machine", "machine"
    ]
}

# Converte para DataFrame do pandas
df = pd.DataFrame(data)

# Converte cada senha para um vetor de features
X = [extract_features(pw) for pw in df["password"]]
y = df["label"]  # Rótulos: human ou machine

# Divide os dados em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ====== Modelo de IA ======

# Cria e treina um modelo de floresta aleatória
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avalia o desempenho do modelo
preds = model.predict(X_test)
print("Relatório de Classificação:\n", classification_report(y_test, preds))

# ====== Salva o modelo treinado ======
joblib.dump(model, "pw_model.pkl")

# ====== Modo interativo ======

def predict(pw):
    """
    Carrega o modelo salvo e faz a previsão para a senha informada.
    """
    if not os.path.exists("pw_model.pkl"):
        print("Modelo não encontrado. Execute o treinamento primeiro.")
        return

    model = joblib.load("pw_model.pkl")
    features = extract_features(pw)
    result = model.predict([features])[0]
    print(f"\nSenha: {pw} → Classificada como: {result.upper()}")

# Interface de linha de comando
if __name__ == "__main__":
    while True:
        entrada = input("\nDigite uma senha para análise (ou 'sair'): ")
        if entrada.strip().lower() == "sair":
            break
        predict(entrada)
