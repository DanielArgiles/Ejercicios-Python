"""
En este proyecto para principiantes, aprenderemos cómo implementar el reconocimiento de rostros humanos en tiempo real. 
Gracias al reconocimiento facial / detección de rostros (tecnología de visión por computador) localizamos y visualizamos los rostros humanos en cualquier imágen digital.
El reconocimiento facial tiene importancia en muchos campos como el marketing y la seguridad.


Los clasificadores en cascada Haar son los métodos utilizados para la detección de objetos.
Haar Cascade Classifier es un algoritmo de detección de objetos de Machine Learning o aprendizaje automático utilizado para identificar objetos en una imagen o video y se basa en el concepto de características propuesto por Paul Viola y Michael Jones en su artículo "Detección rápida de objetos utilizando una cascada mejorada de características simples" en 2001.
Con dicho algoritmo, entrenamos una función en cascada con toneladas de imágenes. 
Estas imágenes se dividen en dos categorías: imágenes positivas que contienen el objeto objetivo e imágenes negativas que no contienen el objeto objetivo.

OpenCV proporciona modelos pre-entrenados sobre características de Haar y clasificadores en cascada. 
Estos modelos se encuentran en la instalación de OpenCV. Puede encontrar los archivos XML necesarios en:
/home/<username>/.local/lib/<python-version>/site-packages/cv2/data/
"""


import cv2
import os

# Inicializamos el clasificador
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Aplicamos faceCascade en los marcos de la webcam
video_capture = cv2.VideoCapture(0)
while True:
    
    # Capturamos marco por marco
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Dibujamos un rectángulo alrededor de las caras
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Mostramos el marco resultante
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Publicamos los marcos de captura       
video_capture.release()
cv2.destroyAllWindows()
