# Eai! bem vindo ao meu repositório!
 Aqui vais encontrar tudo sobre máquinas inteligentes com padrão de conhecimento, inlcuindo modelos de treinamento de IA

## Como executar meus códigos:

<details>
  <summary>Sistema de reconhecimento facial</summary>
 <br>
Dentro do diretório onde o arquivo facetracking.py está localizado, você vai criar uma pasta chamada imagens (Lembrando que o treinar_module.py tem que estar no mesmo diretório que o facetracking.py) <br><br>

<img src="https://raw.githubusercontent.com/savyohenriquedev/Machine-learning/refs/heads/main/Sistema%20de%20reconhecimento%20facial/Directory_images.png" alt="Descrição da imagem" width="200"><br><br>

Depois, dentro da página imagens, você vai criar uma pasta no modelo seu_nome_idade_cargo (é possível criar quantas forem possíveis existir)<br><br>

<img src="https://raw.githubusercontent.com/savyohenriquedev/Machine-learning/refs/heads/main/Sistema%20de%20reconhecimento%20facial/Directory_images_source_1.png" alt="Descrição da imagem" width="700"><br><br>

E dentro dessa(s) pasta(s), você vai colocar fotos (recomendo colocar pelo menos 10 fotos em ângulos diferentes do rosto) do indivíduo pela qual se indentifica a pasta<br><br>

<img src="https://raw.githubusercontent.com/savyohenriquedev/Machine-learning/refs/heads/main/Sistema%20de%20reconhecimento%20facial/Directory_images_source.png" alt="Descrição da imagem" width="400"><br><br>

Feito isso, você vai executar o treinar_module.py primeiro, e depois que finalizar e criar 2 arquivos, você executa o facetracking.py e daí a máquina já foi treinada para reconhecer o rosto.<br><br>

_Vale lembrar que o treinar_module.py serve para treinar a máquina a reonhecer seu rosto, toda vez que você atualizar as fotos da pasta images ou criar um novo usuário, você tem que executá-lo. Os dados processados para treinamento
do módulo ficam salvos nos novos arquvios knn.knl criados_
 
</details>

<details>
  <summary>Avaliador de senhas diferidas se feitas por um robô ou um humano</summary>
 <br>
Este é um projeto simples de inteligência artificial feito só por curiosidade.

A ideia aqui é treinar uma IA para tentar adivinhar se uma senha foi criada por uma pessoa 
ou se foi gerada automaticamente (como aquelas senhas aleatórias que alguns sites criam).

Parece besteira, mas tem lógica: senhas feitas por humanos geralmente seguem padrões, 
como nomes, datas ou sequências do teclado (ex: "joao123", "senha2024", "abc123").
Já as senhas geradas por máquina são bem mais caóticas (ex: "G7#pLx9!Q").

A IA aprende a reconhecer esses padrões com base em alguns critérios simples:
- Quantidade de letras, números e símbolos
- Uso de maiúsculas e minúsculas
- E um pouco de estatística (a chamada "entropia", que mede o quanto a senha parece aleatória)

No final, é só um projetinho leve, mas serve bem como exemplo de:
- Extração de features manuais
- Treinamento de modelo com scikit-learn
- Entrada interativa via terminal
- Classificação binária (humano vs. máquina)

*Não é um sistema de segurança reaaaaal nem tem base em grandes bancos de dados.
É só um experimento para brincar com IA de forma prática e direta.*
</details>
