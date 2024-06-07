# Evidencia_Final_Compiladores
## Reglas Gramáticales usadas

Estas reglas grámaticales se pueden encontrar en [parser.out](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/codigos/parser.out)
```
Rule 0     S' -> assignment
Rule 1     assignment -> VARIABLE EQUAL expression
Rule 2     assignment -> expression
Rule 3     factor -> listValue
Rule 4     assignment -> VARIABLE EQUAL flow
Rule 5     flow -> VARIABLE CONNECT flow_functions
Rule 6     flow_functions -> flow_function_call CONNECT flow_functions
Rule 7     flow_functions -> flow_function_call
Rule 8     flow_function_call -> VARIABLE LPAREN params RPAREN
Rule 9     expression -> expression PLUS term
Rule 10    expression -> expression MINUS term
Rule 11    expression -> term
Rule 12    expression -> string_def
Rule 13    string_def -> STRING
Rule 14    term -> exponent
Rule 15    term -> term TIMES exponent
Rule 16    term -> term DIV exponent
Rule 17    exponent -> factor
Rule 18    exponent -> factor EXP factor
Rule 19    factor -> LBRACKET params RBRACKET
Rule 20    listValue -> VARIABLE LBRACKET expression RBRACKET
Rule 21    factor -> NUMBER
Rule 22    factor -> VARIABLE
Rule 23    factor -> LPAREN expression RPAREN
Rule 24    factor -> function_call
Rule 25    function_call -> VARIABLE LPAREN RPAREN
Rule 26    function_call -> VARIABLE LPAREN params RPAREN
Rule 27    params -> params COMMA expression
Rule 28    params -> expression
```
## Descripción de Funciones implementadas como herramientas y accesorios a la gramática

Las siguientes funciones se encuentran en nuestro archiivo [library.py](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/codigos/library.py)
### Load Image
- Nuestro load image ayuda a tener la ruta en donde se encuentra una imagen
```
def load_image(image_path):
    path = image_path.strip()
    return cv2.imread(path)
```
### Load Image Gray
- Para poder realizar el histograma, se realizo un código para convertir una imagen en escala de grises
```
def load_image_gray(image_path):
    path = image_path.strip()
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```
### Show Histogram
- Código para realizar histograma de una imagen en nuestro translator
```
def show_histogram(image):
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title("Histogram for gray scale picture")
    plt.xlabel("Pixel Value")
    plt.ylabel("Number of Pixels")
    plt.show()
```
### Show Image
- Funcion para mostrar una imagen 
```
def show_image(image):
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image
```
### Search cv2
- Funcion para poder usar las funciones que ofrece cv2
```
def search_cv2(function_name):
    try:
        return getattr(cv2, function_name)
    except:
        pass
    return None
```
### Search Numpy
- Funcion para poder usar las funciones que ofrece numpy
```
def search_numpy(function_name):
    try:
        return getattr(np, function_name)
    except:
        pass
    return None
```
### Gen Vector
- Funcion para generar vectores
```
def gen_vector(*args):
    return np.array(args)
```
### Print File
- Función para imprimir el contenido de un file 
```
def print_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())
```
### Apply Watershed
- Funcion de Watershed Algorithm
```
def apply_watershed(image_path):
    # Load the image
    img = cv2.imread(image_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    show_image(img)
```

## Demostración de una o varias expresiones y el árbol de sintaxis abstracto demostrando

```
```
Para correr nuestro traductor correr en tu terminal la versión de python y el archivo [translator.py](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/codigos/translator.py) que se encuentra en la carpeta de codigos
```
python3 translator.py
```

### Aceptar archivos y ejecutar el contenido
Input:
```
load test.txt
```
Output
```
Command in file:  a=3
Result 3
Command in file:  b=2
Result 2
Command in file:  c=sumAB(a,b)
Result 5
End of file
```
Grafo
<br>
![Grafo 1.1](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo1_1.png)
<br>
![Grafo 1.2](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo1_2.png)
<br>
![Grafo 1.3](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo1_3.png)
### Implementar flujos de funciones (->) que solo reciban la imagen como parámetro (como show_images)
Lo que nos servirá este código es poder conectar varias funciones y que se ejecute facilmente, ya que ayuda a cambiar un elemnto consecutivamente sin tener que usar varias lineas de código. Por ejemplo si quisieramos editar una imagen con funciones como blur() luego lines() y finalizar con edges() se podría realizar conectando las funciones con una flecha "->"
<br>
Input:
```
b=a->sumAB(2)->sumab(4)
```
Output
```
{'type': 'FLOW_FUNCTION_CALL', 'label': 'ff_sumAB', 'value': 'sumAB', 'counter': 2}
{'type': 'FLOW_FUNCTION_CALL', 'label': 'ff_sumab', 'value': 'sumab', 'counter': 5}
Result None
```
Grafo
<br>
![Grafo 2](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo2.png)

### Aceptar None como valor de la gramática para inicialización de variables
Sí quisieramos declarar una variable como None, nuestro traductor ya tiene el valor en nuestra tabla de simbolos.
<br>
Input y Output:
```
>a=None
Result None
>a
Result None
```
Grafo
<br>
![Grafo 3](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo3_1.png)
<br>
![Grafo 3](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo3_2.png)

### Aceptar cualquier función de numpy para manejo de matrices como np.where, np.mean, np.std. Al menos 9 de ellas.
Sí se quisiera insertar una funcion de la libreria numpy, el traductor lo pudiera realizar con el código en la libreria [Search Numpy](#search-numpy)
<br>
Input and Ouput:
```
>a=tuple(1,2,3,4)
Result [1 2 3 4]
>mean(a)
Result (2.5, 0.0, 0.0, 0.0)
```
Grafo
<br>
![Grafo 4](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo4_1.png)
<br>
![Grafo 4](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo4_2.png)

### Implementación de visualización de histogramas con opencv
Con la ayuda de [OpenCV](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html) se implemento una función para poder crear un histograma con los valores de los pixeles de una imagen con las siguientes funciones de nuestra libreria:
- [Show Histogram](#show-histogram)
- [Load Image Gray](#load-image-gray)
<br>
Input and Ouput:
```
>b=loadGray("image.png")
Result [[72 72 70 ... 87 87 87]
 [72 72 70 ... 87 87 87]
 [70 70 70 ... 87 87 87]
 ...
 [28 21 28 ... 53 53 53]
 [21 21 21 ... 46 53 56]
 [21 21 21 ... 46 53 56]]
>histogram(b)
Result None
```
Grafo
<br>
![Grafo 5](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo5_1.png)
<br>
![Grafo 5](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo5_2.png)
<br>
![Output 5](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/output5_1.png)
<br>
![Output 5](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/output5_2.png)

### Aceptar el manejo de listas: A[10]=12, A[0:12] = 20, A[3][4] = 4
<br>
Al momento que nuestro traductor detecte que se introducio una lista se pueden realizar operaciones con los contenidos de estas

Input and Ouput:
```
>x=[13,5,67,23,90]
Result [13, 5, 67, 23, 90]
>x
Result [13, 5, 67, 23, 90]
>x[4]
Result 90
>x[4]*x[1]
Result 450
```
Grafo
<br>
![Grafo 6](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo6_1.png)
<br>
![Grafo 6](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo6_2.png)
<br>
![Grafo 6](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo6_3.png)
<br>
![Grafo 6](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo6_4.png)

### Implementación de un algoritmo complejo como herramienta en el lenguaje WaterShed

La función apply_watershed en Python carga una imagen, la convierte a escala de grises y aplica un binarizado inverso para separar el primer plano del fondo. Luego, elimina el ruido mediante operaciones morfológicas las cuales manipulan la forma de los objetos mediante transformaciones. Marca las regiones desconocidas y aplica el algoritmo de watershed para segmentar la imagen, resaltando las fronteras detectadas en azul, y mostrar la imagen segmentada. Usamos la funcion que se encuentra en nuestra libreria [Apply Watershed](#apply-watershed)
<br>
Input:

```
watershed("coins.png")
```
Output:
<br>
![output 7](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/output7.png)
<br>
Grafo:
<br>
![Grafo 7](https://github.com/AnYelg/Evidencia_Final_Compiladores/blob/main/img/grafo7.png)
