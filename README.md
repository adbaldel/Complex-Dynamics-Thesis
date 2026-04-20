# **Tesis de Grado: Dinámica Compleja y Funciones Racionales (Español)**

## **Descripción**

Este repositorio contiene el código, las figuras y los archivos fuente en LaTeX de mi Trabajo de Fin de Grado (TFG) sobre **Dinámica Compleja**, centrado específicamente en la iteración de funciones racionales en la esfera de Riemann.
El objetivo principal de la tesis es construir una base sólida de los conceptos topológicos y analíticos necesarios (siguiendo el libro *Iteration of Rational Functions* de Alan F. Beardon) para finalmente enunciar y comprender el **Teorema de los Dominios no Errantes** (No Wandering Domains Theorem) de Sullivan.
El código en Python de este repositorio se utiliza para explorar estos conceptos de forma numérica y para generar los fractales de alta calidad (conjuntos de Julia, cuencas de Fatou, etc.) que se incluyen en la memoria escrita.

## **Estructura del Repositorio**

* **code/**: Contiene todos los archivos de programación en Python.
  * **notebooks/**: Cuadernos de Jupyter que ilustran ejemplos específicos (ej. transformaciones de Möbius, la dinámica de $z \\mapsto z^2$).
  * **src/**: Módulos principales de Python (dynamics.py, utils.py) que contienen las funciones y clases reutilizables para calcular los sistemas dinámicos.
* **figures/**: Directorio de salida para los gráficos e imágenes de fractales generados (ej. cuencas del método de Newton) que se usan en el documento LaTeX.
* **report/**: Contiene el código fuente en LaTeX, la bibliografía (refs.bib) y el PDF compilado de la memoria de la tesis (\~50 páginas).

## **Requisitos y Configuración**

Para ejecutar el código e interactuar con los cuadernos de Jupyter, necesitarás un entorno de Python. La forma más sencilla de configurarlo es usando Conda.

1. **Clonar el repositorio:**
   ```bash
   git clone \[https://github.com/adbaldel/Rational-Dynamics-Thesis.git\](https://github.com/adbaldel/Rational-Dynamics-Thesis.git)
   cd Rational-Dynamics-Thesis
   ```

3. **Crear un entorno de Conda e instalar las dependencias:**
   Usando el archivo requirements.txt proporcionado, ejecuta los siguientes comandos:
   ```bash
   conda create \--name dynamics\_env python=3.10
   conda activate dynamics\_env
   pip install \-r requirements.txt
   ```

## **Uso**

* **Explorar el Código:** Activa tu entorno de conda e inicia Jupyter para ver los ejemplos interactivos:
  jupyter notebook

  Navega hasta el directorio code/notebooks/ para abrir los archivos .ipynb.
* **Compilar la Memoria:** Navega hasta el directorio report/ y utiliza tu compilador de LaTeX preferido (ej. pdflatex o latexmk) sobre el archivo main.tex.

# **Bachelor's Thesis: Complex Dynamics & Rational Functions**

## **Description**

This repository contains the code, figures, and LaTeX source files for my Bachelor's thesis on **Complex Dynamics**, specifically focusing on the iteration of rational functions on the Riemann sphere.
The main objective of the thesis is to build a solid foundation of the required topological and analytic concepts (following Alan F. Beardon's *Iteration of Rational Functions*) to ultimately state and understand Sullivan's **No Wandering Domains Theorem**.
The Python code in this repository is used to explore these concepts numerically and to generate the high-quality fractals (Julia sets, Fatou basins, etc.) featured in the written report.

## **Repository Structure**

* **code/**: Contains all the Python programming files.
  * **notebooks/**: Jupyter notebooks demonstrating specific examples (e.g., Möbius transformations, the dynamics of $z \\mapsto z^2$).
  * **src/**: Core Python modules (dynamics.py, utils.py) containing the reusable functions and classes for computing dynamical systems.
* **figures/**: Output directory for generated plots and fractal images (e.g., Newton's method basins) used in the LaTeX report.
* **report/**: Contains the LaTeX source code, bibliography (refs.bib), and compiled PDF of the thesis document (\~50 pages).

## **Requirements & Setup**

To run the code and interact with the Jupyter notebooks, you will need a Python environment. The easiest way to set this up is using Conda.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adbaldel/Rational-Dynamics-Thesis.git
   cd Rational-Dynamics-Thesis
   ```

3. **Create a Conda environment and install dependencies:**
   Using the provided requirements.txt file, run the following commands:
   ```bash
   conda create \--name dynamics\_env python=3.10
   conda activate dynamics\_env
   pip install \-r requirements.txt
   ```

## **Usage**

* **Exploring the Code:** Activate your conda environment and start Jupyter to view the interactive examples:
  ```bash
  jupyter notebook
  ```
  Navigate to the code/notebooks/ directory to open the .ipynb files.
* **Compiling the Report:** Navigate to the report/ directory and use your preferred LaTeX compiler (e.g., pdflatex or latexmk) on main.tex.
