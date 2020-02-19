import cv2 
import numpy as np
import pandas as pd
import argparse
# Instalamos las librerías necesarias desde cmd: pip install opencv-python numpy pandas 

# Creamos un analizador de argumentos de la línea de comando, con la librería argparse para la ruta de la imagen.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

# Leemos la imagen con OpenCV (librería de visión artificial)
img = cv2.imread(img_path)

# Declaramos variables globales 
clicked = False
r = g = b = xpos = ypos = 0

# Leemos el archivo csv y lo cargamos en el DataFrame de Pandas. Asignamos un nombre a cada columna. 
index=["color","color_name","hex","R","G","B"] # Ejemplo: arylide_yellow,"Arylide Yellow",#e9d66b,233,214,107
csv = pd.read_csv('colors.csv', names=index, header=None)

# Función para calcular la distancia mínima de todos los colores y obtener el color más coincidente
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

# Función para obtener coordenadas x,y tras el doble clic del mouse
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
       
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_function)

# Bucle infinito
while(1):

    cv2.imshow("image",img)
    if (clicked):
   
        # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 llena todo el rectángulo
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        # Creamos una cadena de texto para mostrar (nombre de color y valores RGB)
        text = getColorName(r,g,b) +', Red='+ str(r) +  ', Green='+ str(g) +  ', Blue='+ str(b)
        
        
        #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text,(50,50),3,0.6,(255,255,255),1,cv2.LINE_AA)

        # Para colores muy claros, mostraremos el texto en color negro
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),3,0.6,(0,0,0),1,cv2.LINE_AA)
            
        clicked=False

    # Romper el bucle infinito cuando el usuario presione la tecla 'esc'
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()

"""" 
Ejecutamos el archivo python desde cmd : python color_detection.py -i <ruta de la imagen a analizar: color_animales.jpg>
Hacemos Doble clic en la ventana para conocer el nombre del color
"""
