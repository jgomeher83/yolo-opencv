
"""
Reconocer objetos en video con OpenCV deep learning library
Algoritmo:
1. Leer Video
2. Cargar la red neuronal de YOLOV3
3. Obtener Blob 
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
1. Leer Video
"""

#Leer video
video = cv2.VideoCapture('videos/overpass.mp4')

#Inicializando variable writer para escribir en los frames procesados
writer = None

#preparando variables para dimensiones espaciales
h, w = None, None

"""
Cargar la red neuronal de YOLOV3
"""

# Carcando el COCO-Dataset

with open(r'yolo-coco-data\coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
print('List with labels names:')
print(labels)

#Cargandop archivos YOLO
#Pesos pre entrenados en COCO- Dataset
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')


#Obteniendo lista con los nombres de las capas da la red YOLO V3
layers_names_all = network.getLayerNames()

# # Punto de control
print()
print(layers_names_all)


#Obteniendo solo los nombres de las capas de salida que se necesitan de yolo
# with function that returns indexes of layers with unconnected outputs
#La funcion retorna los indices de las capas
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
print()
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# probabilidad minima
probability_minimum = 0.5

# threshold para non-maximum suppression
threshold = 0.3

# Generando colores de la longitud del dataset
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Punto de control
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]

"""
Para leer todos los frame del video es necesario un loop
mientras aun hayan frames
"""

# variable para contrar los frames
f = 0

#variable para el tiempo
t = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Dimensiones espaciales del video
    if w is None or h is None:
        h, w = frame.shape[:2]

    """
    3. Obtener Blob 
    """

    #La funcion 'cv2.dnn.blobFromImage' va a retornar un blob de 4 dimensiones
    #de la imagen ingresada despues de la normalizacion
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

   
    """
    4. Implementar forward Pass
    """

    #pasarle a la red el blob 
    #utilizando solo las capas necesarias
    #medir el tiempo de pasar el blob por la red
    network.setInput(blob) 
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Incrementando contadores
    f += 1
    t += end - start

    # mostrar tiempo de procesamiento para cada frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))


    """
    5. Obtener las cajas de detecciones
    """

    # definicion de listas necesarias para guardar
    #informacion de detecciones
    bounding_boxes = []
    confidences = []
    class_numbers = []

    """
    Es necesario recorrer las capas de salida
    es decir la variable 'output_from_network'
    y luego recorrer para cada capa de salida
    todas las detecciones que tuvo
    """
    for result in output_from_network:
        #recorriendo todas las detecciones de la capa de salida
        for detected_objects in result:
            #obtener la probabilidad de las 80 clases del COCO-Dataset
            scores = detected_objects[5:]
            #Luego de tener las probabilidades, con la ayuda de numpy obtenemos 
            #el índice de la clase con mayor probabilidad
            class_current = np.argmax(scores)
            # obtenemos el valor de esa probabilidad
            confidence_current = scores[class_current]

              #ELiminando predicciones que no cumplen el mínimo determinado en la 
              #constante probability_minimum
            if confidence_current > probability_minimum:
              
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Desmpaquetamiento del arreglo para obtener las coordenadas de las cajas
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Se añaden los resultados a las listas
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current)) #probabilidad de la clase detectada
                class_numbers.append(class_current) #indice de la clase detectada


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

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()

            # # # Punto de control
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127] 
            #el color cero del arreglo de colores

            # Dibujar la caja de detección en el frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Clase y probabilidad detectada
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Dibujando el texto de la caja en el frame
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

  

    # Inicializamos el writer
    if writer is None:
        #Escogemos el codec ver documentacion OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        #Ver documentacion parta VideoWriter
        writer = cv2.VideoWriter('videos/pedestrian_out.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    #Escribiendo el video de salida frame por frame
    writer.write(frame)




# Impiriendo datos finales
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# release
video.release()
writer.release()

