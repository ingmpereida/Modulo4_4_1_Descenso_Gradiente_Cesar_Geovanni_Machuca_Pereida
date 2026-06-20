# Descenso por Gradiente desde Cero (Linear Regression from Scratch)

Este repositorio contiene la implementación matemática y en código del algoritmo de **Descenso por Gradiente** aplicado a un modelo de **Regresión Lineal Simple**, desarrollado en un entorno de Jupyter Notebook (`.ipynb`) compatible con Google Colab.

---

## 🏛️ Información Institucional

* **Institución:** Centro de Investigación y Estudios Avanzados del IPN (Cinvestav) - Unidad Guadalajara
* **Módulo:** Módulo 4: Deep Learning (Actividad en clase)
* **Profesor:** Dr. German Alonso Pinedo Díaz
* **Alumno:** César Geovanni Machuca Pereida

---

## 🚀 Descripción del Proyecto

El notebook se enfoca en resolver el problema de ajustar una línea recta de la forma:

$$y = \phi_0 + \phi_1 x$$

A partir de un conjunto de datos bidimensional (12 pares de puntos $\{x_i, y_i\}$), el algoritmo optimiza iterativamente los parámetros del modelo:
* $\phi_0$: Intercepto (sesgo/bias).
* $\phi_1$: Pendiente (peso/weight).

El objetivo principal de esta actividad es comprender la mecánica interna de la optimización matemática en Machine Learning, construyendo los componentes clave de forma vectorial sin depender de librerías de alto nivel como Scikit-Learn.

---

## 🛠️ Estructura del Código y Ejercicios Implementados

El código está estructurado en módulos clave que resuelven etapas específicas del proceso de optimización:

### 1. Función de Pérdida (Suma de Cuadrados)
Calcula la discrepancia total entre las predicciones del modelo y los valores reales observados mediante el Error Cuadrático Total:

$$L(\phi)=\sum_i(f(x_i,\phi)-y_i)^2$$

### 2. Cálculo del Gradiente Analítico
Determina la dirección del máximo crecimiento del error mediante las derivadas parciales respecto a cada parámetro de forma vectorial:

$$\nabla_\phi L = \begin{bmatrix} \frac{\partial L}{\partial \phi_0} \\ \frac{\partial L}{\partial \phi_1} \end{bmatrix} = \begin{bmatrix} 2 \sum_i (\phi_0 + \phi_1 x_i - y_i) \\ 2 \sum_i (\phi_0 + \phi_1 x_i - y_i) \cdot x_i \end{bmatrix}$$

### 3. Verificación Numérica (Diferencias Finitas)
Se incluye un bloque de control de calidad para validar que las derivadas analíticas sean correctas aproximando el gradiente mediante diferencias finitas:

$$\frac{\partial L}{\partial \phi_0}\approx \frac{L(\phi_0+\delta,\phi_1)-L(\phi_0,\phi_1)}{\delta}$$

*Nota: Se implementó la extracción explícita de escalares mediante `.item()` en NumPy para evitar advertencias de depreciación (`DeprecationWarning`).*

### 4. Búsqueda Lineal (*Line Search*)
Implementa un algoritmo de exploración acotada en una dimensión para estimar dinámicamente la tasa de aprendizaje ($\alpha$), asegurando un descenso fluido hacia el mínimo global sin oscilaciones ni divergencia.

### 5. Paso de Descenso e Iteración
Aplica de forma iterativa la regla de actualización de parámetros:

$$\phi_{t+1}=\phi_t-\alpha \nabla_\phi L(\phi_t)$$

---

## 📋 Tecnologías Utilizadas

* **Python 3**
* **NumPy**: Operaciones vectorizadas y manipulación de matrices de datos.
* **Matplotlib**: Visualización dinámica del ajuste de la recta y graficación de curvas de nivel en la superficie de pérdida.

---

## 📊 Visualizaciones Incluidas

Al ejecutar el flujo completo del notebook, se generan dos tipos de gráficas interactivas:
1.  **Ajuste del Modelo en Vivo:** Muestra cómo la recta magenta rota y se desplaza iteración tras iteración hasta acoplarse perfectamente a la tendencia de los puntos azules de entrenamiento.
2.  **Contorno de Pérdida:** Un mapa topográfico bidimensional donde se aprecia de forma visual el camino en zigzag o curva (línea verde) que describen los parámetros desde una zona de alto error hacia el centro oscuro del valle elíptico (mínimo global).

---

## 💡 Ejemplo de Aplicación en la Vida Real

Este algoritmo matemático exacto puede ser transferido directamente a problemas de estimación industrial. 

**Caso práctico: Predicción de Consumo Eléctrico en Servidores Cloud**
* **Variable $x$:** Porcentaje de carga de trabajo de la CPU en un servidor.
* **Variable $y$:** Consumo de energía en Watts.
* **Parámetro $\phi_0$ (Intercepto):** Energía base que consume el servidor encendido estando en reposo (0% de CPU).
* **Parámetro $\phi_1$ (Pendiente):** Factor de incremento de Watts por cada unidad porcentual de esfuerzo de procesamiento.

Al entrenar este modelo con datos históricos, el descenso por gradiente encuentra la combinación con menor error, permitiendo a los administradores de infraestructura predecir costos operativos y demandas de enfriamiento ante picos de tráfico en tiempo real.
