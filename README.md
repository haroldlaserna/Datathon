# Datathon
### Mediante un $set$ de datos con las siguientes columnas:

- ID: identificador del registro de orden (valor entero).
- Warehouse_block: Almacén de distribución de donde salió la orden (A a F).
- Mode_of_Shipment: Medio de transporte (Flight, Road, Ship).
- Customer_care_calls: Número de llamadas a atención al cliente que hubo por esa orden. (valores enteros del 2 al 7)
- Customer_rating: Puntaje del cliente (valores enteros 1 al 5).
- Cost_of_the_Product: Costo del producto (valor numérico entero de 96 a 310).
- Prior_purchases: Número de compras previas realizadas por el cliente (valor numérico entero de 2 a 10).
- Product_importance: Nivel de importancia del producto (low, medium, high).
- Gender: Género del comprador (F, M).
- Discount_offered: Porcentaje de descuento ofrecido por esa compra (valor numérico entero de 1 a 65):
- Weight_in_gms: Peso del paquete de la orden, en gramos (valor numérico entero de 1001 a 7846).
- Reached.on.Time_Y.N: Información sobre la llegada del paquete a destino (1 si llegó a tiempo, 0 si no llegó a tiempo).

### Intentar generar un modelo con **SKLEARN**, el cual prediga lo máximo posible  si un paquete llegó a tiempo o no.

## el análisis del modelo se encuentra en el notebook de jupyter notebook llamado **main.ipynb**

### **Pipeline**: Nos ayuda a generar un proceso que debe seguir nuestro algoritmo de forma fácil.

### **PCA (Principal Component Analysis)**: Dado una matriz de covarianza obtenido del set de datos mediante autovalores y autovectores, es encontrar una nueva matriz covariante con valores diferentes a cero únicamente en la traza. con esto se encuentran autovectores ortogonales los cuales nos ayuda a transformar nuestros datos. Esto con el objetivo de encontrar nuevas dimensiones, las cuales podrán contener más información que con la matriz de covarianza original. Dado que alguna dimensión podrá contener más información, esto nos ayuda reducir la dimensionalidad, intentando que no se pierda la mayor de informacioón posible.

### **ColumnTransformer**: Dado el caso que se desea operar sobre ciertos columnas y no sobre todas como pasa con el pipeline O también si se desea hacer procesos paralelamente es util usar este objeto para hacer estos procesos.
### **GridSearchCV (Búsqueda exhaustiva en cuadrícula)**: La búsqueda de cuadrícula proporcionada por GridSearchCVgenera exhaustivamente candidatos a partir de una cuadrícula de valores de parámetros especificados con el param_grid parámetro.

### **StandardScaler**: Estandariza una columna o set de datos suponiendo una distrubución normal de los datos dado por la fórmula:
$$x'_{i}=\frac{x_{i}-\mu}{\sigma},$$
### siendo $x_{i}$ el valor original y $x_{i}'$ el valor estandarizado. Es de notar que $i$ es el i-ésimo dato de un conjunto de datos de tamaño $N$.
### Donde $\mu$ es la media aritmética y $\sigma$ es la desviación estándar, para este caso denotadas de la siguiente forma: 
$$\mu=\frac{1}{N}\sum_{i=1}^{N}x_{i}$$
$$\sigma²=\frac{1}{N}\sum_{i=1}^{N}(x_{i}-\mu)^2$$
### **OneHotEncoder**:  transforma cada característica categórica con $n$ categorias posibles en $n$ características binarias, siendo una de ellas 1 y todas las demás 0.
### **PolynomialFeatures**: Esta variable mezcla columnas de con un grado polinomial escogido. por ejemplo: sea un set de datos $Y$ con dos columnas $y_{1}$ y $y_{2}$ vamos a generar nuevas columnas con la combinación de las dos. Escogemos el grado de polinomio: $2$ las nuevas columnas del set de datos $Y$ son $1$, $y_{1}$, $y_{2}$, $y_{1}y_{2}$, $y_{1}^2$ y $y_{2}^2$, bajo esta logica notamos que rige desde el teorema multinomial sumado una unidad, es decir:

$$polinomio = (y_{1}+y_{2}+...+y_{p})^M+1$$

$$polinomio = 1+\sum_{i_{1}+i_{2}+...+i_{p}=M}\binom{M}{i_{1}, i_{2},...,i_{p}}\prod_{1<t<p}x_{t}^{k_{t}}$$
### siendo $p$ la cantidad de columnas, el grado del polinomio $M$, $i_{t}$las combinaciones de enteros no negativos posibles menores a $M$.
### **confusion_matrix**: Genera una matriz de confusión minimo con 4 elementos: Verdadero Positivo, Falso Positivo, Falso Negativo y Verdadero Negativo.
### **ConfusionMatrixDisplay**: Genera un mapa de calor minimo con 4 píxeles de la matriz de confusión.
### **classification_report**: Genera una tabla en pantalla con ciertas relaciones estadisticas importantes a la hora de generar un modelo de clasificación. Dichas estadisticas importantes son:
1. Presición: Denota la fracción de instancias relevantes entre las instancias recuperadas es decir:

$$presición=\frac{verdadero\_positivo}{verdadero\_positivo+falso\_positivo}$$

2. Recuperación: Denota la fraccipon de instancias relevantes que se recuperaron, es decir:

$$recuperacion=\frac{verdadero\_positivo}{verdadero\_positivo+falso\_negativo}$$

3. puntuación F: Denota la media armonica de la recuperación y la precisión, es decir:

$$F=2*\frac{presicion*recuperacion}{presicion+recuperacion}$$

3. Accuracy: muestra la presición total entre la certeza de obtener buenos datos de verdaderos positivos y verdaderos negativos teniendo en cuenta el error de obtener falsos positivo y falsos negativos, es decir:

$$Accuracy = \frac{verdadero\_positivo+verdadero\_negativo}{verdadero\_positivo+verdadero\_negativo+falso\_positivo+falso\_negativo}$$

Si $falso\_positivo$ y $falso\_negativo$ son igual a cero, entonces se tiene que:

$$Accuracy = \frac{verdadero\_positivo+verdadero\_negativo}{verdadero\_positivo+verdadero\_negativo}$$
$$Accuracy = 1$$
### **train_test_split**: Este método puede partir los datos dependiendo un corte posible lo mas uniforme posible.
### **MLPClassifier**: Método de Backpropagation para poder relacionar los datos con unos parametros dados así poder predecir con distintos datos en el futuro.

### $Backpropagation$ es un método el cual mediante datos de entrada se entrena una red neuronal se compara su salida con datos que se esperan que sean los predecidos y luego se va para atrás para calibrar los hiperparámetros. La siguiente imagen es una construcción de una red neuronal.
![alt text](images/backpropagation.jpg)

### Donde cada circunferencia denota una neurona, $w_{ij}$ denota cada peso de cada neurona, $b_{j}$ los bias. Estos dos parámetros anteriores son los que se van calibrando mediante el backpropagation. Cada columna de neuronas es llamada capa oculta. Cada dato de entrada $z_{i}$ a una neurona correspondiente es multiplicada por su peso correspondiente $w_{ij}$ siendo $j$ la $j$-esima neurona y sumandole el parámetro bias, como si fuera una función con tendencia linea. luego de todos, el procedimiento con todos los $z_{i}$ que interactuan con la nuerona, dicho resultado es sumado, dando como resultado la funcion $a_{j}$ como se muestra en la imagen. para finalmente dicha función entrar en una función $h$ llamada función de activación. Dicho mapeo corresponde a:
$$h:\mathbb{R} \to \mathbb{R}$$
### Es decir que convierte el valor $a_{j}$ en un nuevo valor del mismo espacio de los reales, dicho mapeo convierte $a_{j}$ en un nuevo $z_{j}$ que interactuará con la siguiente capa. Tal que la salida última es una mapeo progresivo constante con la función de activación.

### en sklearn existen 4 tipos de funciones de activacion:
### 1. Función identidad: No modifica el valor de $a_{j}$ es decir que la función de identidad es:
$$f(x)=x$$
### 2. Función logistica: Modifica el valor de $a_{j}$ y lo mapea en el espacio $[-1,1]$, está definida de la siguente forma:
$$f(x)=\frac{1}{1+e^{-x}}$$
### 3. Función tangente hiperbólica: Función que mapea también en el espacio $[-1,1]$, está definida de la siguiente forma:
$$f(x)=tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
### 4. ReLU: FUnción que mapea $a_{j}$ al espacio $[0,\infty]$ definida de la siguente forma:
$$
    f(x)= 
\begin{cases}
    x,& \text{if } x > 0\\
    0,              & \text{para otros casos.}
\end{cases}
$$

### Para la actualización de los parametros $w_{ij}$ y $b_{j}$ que intentan minimizar el error $\epsilon$ correspondiente al error medio cuadrático que puede ser derivado desde el $Likelihood$ tomando que los errores se comportan como una distribución normal, se procede mediante métodos de descenso por gradiente.

### Tenemos en cuenta que para minimizar el error tenemos que actualizar los parámetros y es posible por descenso por gradiente de la siguente forma:
$$\theta_{t+1}= \theta_{t} - \alpha \nabla \epsilon  $$
### siendo $\theta$ la matriz con los hiperparámetros de la red neuronal y $\alpha$ la rata de aprendizaje.
### para mejorar la minimización usaremos el metodo ADAM de descenso por gradiente definido de la siguente forma:
$$m_{t}=0.9m_{t-1}+(1-0.9)\nabla \epsilon$$

$$v_{t}=0.9v_{t-1}+(1-0.999)(\nabla \epsilon)²$$

$$\hat{m}_{t}=\frac{m_{t}}{1-0.9^t}$$

$$\hat{v}_{t}=\frac{v_{t}}{1-0.999^t}$$

$$\theta_{t+1}= \theta_{t} - \frac{\alpha}{\sqrt{\hat{v}_{t}}+\beta} \hat{m}_t  $$

### Siendo $\beta =10^{-8}$
