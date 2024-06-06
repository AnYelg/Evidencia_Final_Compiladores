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
- Nuestro load image ayuda a tener la ruta en donde se encuentra una imagen
```
def load_image(image_path):
    path = image_path.strip()
    return cv2.imread(path)
```

- Para poder realizar el histograma, se realizo un código para convertir una imagen en escala de grises
```
def load_image_gray(image_path):
    path = image_path.strip()
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```
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
- Funcion para mostrar una imagen 
```
def show_image(image):
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image
```
- Funcion para poder usar las funciones que ofrece cv2
```
def search_cv2(function_name):
    try:
        return getattr(cv2, function_name)
    except:
        pass
    return None
```
- Funcion para poder usar las funciones que ofrece numpy
```
def search_numpy(function_name):
    try:
        return getattr(np, function_name)
    except:
        pass
    return None
```
- Funcion para generar vectores
```
def gen_vector(*args):
    return np.array(args)
```
- Función para imprimir el contenido de un file 
```
def print_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())
```
## Demostración de una o varias expresiones y el árbol de sintaxis abstracto demostrando

