# **Trabajo de Fin de Grado: Dinámica Compleja y Funciones Racionales**

## **Descripción**

Este repositorio contiene el código, las figuras y los archivos fuente en LaTeX de mi Trabajo de Fin de Grado (TFG) sobre **Dinámica Compleja**, centrado específicamente en la iteración de funciones racionales en la esfera de Riemann.

El objetivo principal de la tesis es construir una base sólida de los conceptos topológicos y analíticos necesarios (siguiendo el libro *Iteration of Rational Functions* de Alan F. Beardon) para finalmente enunciar y comprender el **Teorema de los Dominios no Errantes** (No Wandering Domains Theorem) de Sullivan.

El código en Python de este repositorio se utiliza para explorar estos conceptos de forma numérica y para generar los fractales de alta calidad (conjuntos de Julia, cuencas de Fatou, etc.) que se incluyen en la memoria escrita.

## **Estructura del Repositorio**

* **python/**: Contiene el proyecto Python, su configuración y sus dependencias.
  * **notebooks/**: Cuadernos de Jupyter que ilustran ejemplos específicos (ej. transformaciones de Möbius, la dinámica de $z \\mapsto z^2$).
    * **colab/**: Versiones de los cuadernos preparadas para Google Colab.
  * **src/complex_dynamics/**: Paquete reutilizable para calcular y visualizar los sistemas dinámicos.
  * **tests/complex_dynamics/**: Pruebas del paquete.
  * **docs/** y **scripts/**: Documentación técnica y herramientas auxiliares.
* **latex/**: Contiene los documentos y recursos LaTeX.
  * **thesis/**: Memoria de la tesis (`thesis.tex`) y bibliografía (`refs.bib`).
  * **beardon_notes/**: Apuntes sobre el libro de Beardon (`notes.tex`).
  * **translated_articles/**: Traducciones de artículos, cada una con su propio `main.tex`.
  * **beamer/**: Material para la presentación.
  * **utils/**: Comandos, entornos y configuración compartidos.
  * **figures/**: Gráficos generados por Python, imágenes de referencia y fuentes TikZ utilizados en los documentos.

## **Requisitos y Configuración**

Para ejecutar el código e interactuar con los cuadernos de Jupyter, necesitarás un entorno de Python. La forma más sencilla de configurarlo es usando Conda.

1. **Clonar el repositorio:**
   ```bash
    git clone https://github.com/adbaldel/Rational-Dynamics-Thesis.git
   cd Rational-Dynamics-Thesis
   ```

2. **Crear un entorno de Conda e instalar las dependencias:**
   El archivo de dependencias está en `python/requirements.txt`. Desde la raíz del repositorio:
   ```bash
    conda create --name dynamics_env python=3.10
    conda activate dynamics_env
    pip install -r python/requirements.txt
    pip install --no-deps -e ./python
   ```

## **Uso**

* **Explorar el Código:** Activa tu entorno de conda e inicia Jupyter desde el directorio de los cuadernos:
  ```bash
  cd python/notebooks
  jupyter notebook
  ```
  Los cuadernos locales usan rutas relativas a `python/notebooks/`; las figuras se guardan en `latex/figures/`.
* **Compilar la Memoria:** Desde la raíz del repositorio:
  ```bash
  cd latex/thesis
  latexmk -pdf thesis.tex
  ```
  Compila los apuntes desde `latex/beardon_notes/` con `latexmk -pdf notes.tex`, y las traducciones desde el directorio de cada artículo con `latexmk -pdf main.tex`.

# **Bachelor's Thesis: Complex Dynamics & Rational Functions**

## **Description**

This repository contains the code, figures, and LaTeX source files for my Bachelor's thesis on **Complex Dynamics**, specifically focusing on the iteration of rational functions on the Riemann sphere.

The main objective of the thesis is to build a solid foundation of the required topological and analytic concepts (following Alan F. Beardon's *Iteration of Rational Functions*) to ultimately state and understand Sullivan's **No Wandering Domains Theorem**.

The Python code in this repository is used to explore these concepts numerically and to generate the high-quality fractals (Julia sets, Fatou basins, etc.) featured in the written report.

## **Repository Structure**

* **python/**: Contains the Python project, its configuration, and its dependencies.
  * **notebooks/**: Jupyter notebooks demonstrating specific examples (e.g., Möbius transformations, the dynamics of $z \\mapsto z^2$).
    * **colab/**: Notebook versions prepared for Google Colab.
  * **src/complex_dynamics/**: Reusable package for computing and visualizing dynamical systems.
  * **tests/complex_dynamics/**: Package tests.
  * **docs/** and **scripts/**: Technical documentation and helper tools.
* **latex/**: Contains the LaTeX documents and resources.
  * **thesis/**: Thesis document (`thesis.tex`) and bibliography (`refs.bib`).
  * **beardon_notes/**: Notes on Beardon's book (`notes.tex`).
  * **translated_articles/**: Article translations, each with its own `main.tex`.
  * **beamer/**: Presentation material.
  * **utils/**: Shared commands, environments, and configuration.
  * **figures/**: Python-generated plots, reference images, and TikZ sources used in the documents.

## **Requirements & Setup**

To run the code and interact with the Jupyter notebooks, you will need a Python environment. The easiest way to set this up is using Conda.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adbaldel/Rational-Dynamics-Thesis.git
   cd Rational-Dynamics-Thesis
   ```

2. **Create a Conda environment and install dependencies:**
   Dependencies are listed in `python/requirements.txt`. From the repository root:
   ```bash
    conda create --name dynamics_env python=3.10
    conda activate dynamics_env
    pip install -r python/requirements.txt
    pip install --no-deps -e ./python
   ```

## **Usage**

* **Exploring the Code:** Activate your conda environment and start Jupyter from the notebook directory:
  ```bash
   cd python/notebooks
   jupyter notebook
  ```
   Local notebooks use paths relative to `python/notebooks/`; figures are saved in `latex/figures/`.
* **Compiling the Report:** From the repository root:
   ```bash
   cd latex/thesis
   latexmk -pdf thesis.tex
   ```
   Compile the notes from `latex/beardon_notes/` with `latexmk -pdf notes.tex`, and translations from each article's directory with `latexmk -pdf main.tex`.
