## RECONOCER OBJETOS EN IMAGEN/VIDEO/CAMARA USANDO YOLOV3/V4 CON OPENCV
<img src="detecciones/deteccion_oficina.png" width="350" height="350" />

<img src="gif/bloggif_5f24d1b132d54.gif" width="350" height="250" />


Reconocer objetos usando la librería OpenCV
## YOLOV3 / YOLOV4 🚀
_Reconocer objetos en imagen/video/camara con OpenCV deep learning library, utilizando dnn, y darknet framework.
Es necesario descargar los pesos para cada versión de YOLO.
* [yolov3.weights](https://pjreddie.com/darknet/yolo) - Descargar yolo.weights
* [yolov4.weights](https://drive.google.com/file/d/1-_-Nwz1RwQqZglKqg-E-04lhWM1RvsaN/view?usp=sharing) - Archivo en drive
### Requisitos📋

Python
Anaconda(recomendado)
OpenCV
Numpy

### Instalación 🔧
1. Poner los pesos en la carpeta yolo-coco-data
2. Descargar Anaconda
* [Anaconda](https://www.anaconda.com/products/individual) - Descargar anaconda
3. Abrir Anaconda Prompt
4. Crear un ambiente virtual

conda create --name go_ahead_env

5. Activar el ambiente

conda activate go_ahead_env

_6. Instalar las librerias necesarias

conda install -c conda-forge opencv


conda install -c anaconda numpy

7. Configurar path de pesos y configuracion
Para correr Yolov3 abrir el script yolo_image.py y modificar path en la linea 56

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg','yolo-coco-data/yolov3.weights')

Para correr Yolov4 abrir el script yolo_image.py y modificar path en la linea 56

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg', 'yolo-coco-data/yolov4.weights')

8. Correr el detector imagenes yolo_image.py en anaconda prompt
navegar hasta la carpeta yolo-opencv y correr el script:

python yolo_image.py

9. Detectar objetos en otras imagenes
Poner la imagen de interes en la carpeta images
Abrir en algun editor el script yolo_image.py y configurar el path en la linea 26

imagen_BGR = cv2.imread('images/nombre_imagen.jpg')

Guardar cambios y correr el script

python yolo_image.py

10. Detectar objetos en video

python yolo-3-video.py

por defecto detectará objetos en el archivo overpass.mp4 ubicado en la carpeta videos
para detectar objetos en otro video, añadirlo a la carpeta videos y configurar path en la linea 28

video = cv2.VideoCapture('videos/name_video.mp4')

Configurar el path de los pesos(paso 7) según la versión de Yolo a utilizar

##¿Cuál versión de yolo funcionó mejor para tu proyecto?
¿Cuál versión de yolo funcionó mejor para tu proyecto?

##GOOD LUCK! Creado por jgomher83 para go_ahead
GOOD LUCK! Creado por jgomeher83 para go_ahead

### Documentacion oficial
* [OpenCV](https://opencv.org/) - OpenCv
