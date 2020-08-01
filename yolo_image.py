"""
Reconocer objetos en imagen con OpenCV deep learning library
Algoritmo:
1. Leer la imagen
2. Obtener Blob
3. Cargar la red neuronal de YOLOV3
4. Implementar forward Pass
5. Obtener las cajas de detección
6. Aplicar Non maximum Supression
7. Dibujar la cajas de deteccion finales

RESULTADO = Ventana con los objetos detectados

"""
#Librerias necesarias
import numpy as np
import cv2
import time

"""
1. Leer imagen
"""
#Leer imagen con OpenCV. Las imagenes son leidas como 
#numpy array. Por defecto OpenCV lee las imagenes 
#en formato BGR
imagen_BGR = cv2.imread('images/oficina.jpg')
h, w = imagen_BGR.shape[:2] #obteniendo de la tupla solo los 2 primeros elementos
"""
fin leer imagen
"""
"""
2. Obtener blob
"""
#La funcion 'cv2.dnn.blobFromImage' va a retornar un blob de 4 dimensiones
#de la imagen ingresada despues de la normalizacion
# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
blob = cv2.dnn.blobFromImage(imagen_BGR, 1 / 255.0, (416, 416), swapRB = True, crop = False)
#Punto de control
# print('Image shape:', image_BGR.shape)  # (511, 767, 3)
# print('Blob shape:', blob.shape)  # (1, 3, 416, 416)
"""
fin obtener blob
"""
"""
3. Cargar la red neuronal  Yolo desde OpenCV
"""
#cargar los labels del archivo coco.names
with open('yolo-coco-data/coco.names') as f:
    #obteniendo los labels de cada linea y ponerlos en una lista
    labels = [line.strip() for line in f]

#Cargando red neuronal YOLOV3 includida en la librería dnn de OpenCV
#Es un detector de objetos ya pre entrenado
# estos dos archivos se pueden descargar de la pagina ppal de yolo
#yolo-coco-data es la carpeta que cotiene los archivos de YoloV3
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')
                        
#obtener todas las capas de la red YOLOV3
layers_names_all = network.getLayerNames()

layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

#establecer probabilidad para la deteccion de objetos
#estos valores se pueden modificar segun las necesidades del proyecto
probability_minimum = 0.5
threshold = 0.3

#Generar colores para las 80 labels
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
"""
fin carga red neuronal YOLOV3
"""

"""
4. Implementar forward pass
"""
#pasarle a la red el blob 
#utilizando solo las capas necesarias
#medir el tiempo de pasar el blob por la red
network.setInput(blob) 
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
print('La detección de objetos demoró {:.5f} segundos'.format(end - start))

"""
5. Obtener las cajas de detecciones
"""
# definicion de listas necesarias para guardar
#informacion de detecciones

bounding_boxes = []
confidences = []
class_numbers = []

#Es necesario recorrer las capas de salida
#es decir la variable 'output_from_network'
#y luego recorrer para cada capa de salida
#todas las detecciones que tuvo

#recorrido de todas las capas después del forward pass
for result in output_from_network:
        #recorriendo todas las detecciones de la capa de salida
    for detected_objects in result:
        #obtener la probabilidad de las 80 clases del COCO-Dataset
        scores = detected_objects[5:] #los primeros 4 elementos del arreglo son las coordenadas de las cajas
        #Luego de tener las probabilidades, con la ayuda de numpy obtenemos el índice de la clase con mayor probabilidad
        class_current = np.argmax(scores)
        # obtenemos el valor de esa probabilidad
        confidence_current = scores[class_current]

    

        # ELiminando predicciones que no cumplen el mínimo determinado en la constante probability_minimum
        if confidence_current > probability_minimum:
    
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            # Desmpaquetamiento del arreglo para obtener las coordenadas de las cajas
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # Se añaden los resultados a las listas
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

"""
6. Aplicar non-maximum suppression
"""

# Implementando la supresión no máxima de cuadros delimitadores dados 
# Con esta técnica, excluimos algunos de los cuadros delimitadores 
# si sus confidencias correspondientes son bajas o si hay otro cuadro 
# delimitador para esta región con mayor confianza

#Las cajas deben tener formato int y las prbabilidades tipo float
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)


"""
7. Dibujar la cajas de deteccion finales
"""



# Checkeo detecciones encontradas
if len(results) > 0:
    # recorriendo los indices de la variable results
    for i in results.flatten():

        # Desmpaquetando las coordenadas de las cajas guardadas en la lista bounding_boxes
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # Preparando color para las cajas de detección
        colour_box_current = colours[class_numbers[i]].tolist()

        # # # Punto de control
        # print(type(colour_box_current))  # <class 'list'>
        # print(colour_box_current)  # [172 , 10, 127]

        # Dibujar la caja de detección en la imagen
        cv2.rectangle(imagen_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        # Clase y probabilidad detectada
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Dibujando el texto de la caja en la imagen
        cv2.putText(imagen_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)





cv2.namedWindow('Detecciones', cv2.WINDOW_NORMAL)
cv2.imshow('Detecciones', imagen_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Detecciones')

