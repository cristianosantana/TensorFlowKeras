# TensorFlowKeras
++++++++++++++++++++++++++++++++++++
Autor do projeto base deste projeto
++++++++++++++++++++++++++++++++++++

Deep Learning Summer School 2018
Hands On TensorFlow and Keras
Sandro Silva Moreira
moreira.sandro@gmail.com

++++++++++++++++++++++++++++
Instruções:
++++++++++++++++++++++++++++

Este exemplo usa uma rede neural Inception com arquitetura pré-treinada (transfer learning) para extrair as características da imagem analisada.

A última camada, de classificação foi treinada para reconhecer alimentos. Este algoritimo conhece as seguintes classes: hot dog, escargots, cannoli, hamburger, french fries, lasagna e grilled salmon.

Para testar, utilize o script: label_image.py

ex:   python3.5 label_image.py [imagem_a_ser_analisada.jpg]

O resultado da classificação será feito mostrando a probabilidade da "imagem_a_ser_analisada.jpg" ser cada uma das classes. As classes de mais alta probabilidade aparece no arquivo resultado.

Este projeto só roda dentro do container lexcrt/debian:tensorflow-keras-v1, então instale o docker e rode o comando a seguir: 'sudo docker run -it -v ~/Workspace://home -p 8888:8888 --rm lexcrt/debian:tensorflow-keras-v1'