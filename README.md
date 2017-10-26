# Project 1 of Pattern Classification and Machine Learning - 2017/2018

[CS-433 PCML](http://isa.epfl.ch/imoniteur_ISAP/!itffichecours.htm?ww_i_matiere=2217650315&ww_x_anneeAcad=2017-2018&ww_i_section=249847&ww_i_niveau=&ww_c_langue=en) - [EPFL](http://epfl.ch)

> Discovery of Higgs Boson using CERN's public dataset. The submission is made to competition platform [kaggle](https://www.kaggle.com/c/epfml-higgs)

## Final Submission Result

* Public leaderboard
  - **TBD** with **TBD%** of accuracy.
* Private Leadeboard
  - **TBD** with **TBD%** of accuracy.

## Project Structure

- `higgs-data`: the CERN's public Higgs-Boson discovery data.
- `report`: report in LaTeX
- `scripts`: all main ML functions and helpers.
- `setup`: Contains readme to give git syntax cheatsheet.

## How to use - Execute using Python Runtime only

1. Ensure that you have python 3 in your machine.
2. Run the run.py to create our final submission file to [kaggle](https://www.kaggle.com/c/epfml-higgs/leaderboard):

  ```bash
  $ cd IST_ML_PROJECT1/
  $ python run.py
  ```

3. To reuse either one of six ML implementations method. Load the implementations.py script and pass the required parameters:

  ```python
  from scripts.implementations.py import [function]
  # Example to run linear regression using gradient descent
  weights, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
  ```

  or import all methods
  ```python
  from scripts.implementations.py import *
  # Example to run least squares with normal equations
  weights, loss = least_squares(y, tx)
  ```

## How to use - Jupyter Notebook

1. Install [Anaconda](https://www.continuum.io/downloads)
2. Clone this repository
3. Run the jupyter-notebook (default is enabled by conda) in terminal:

  ```bash
  $ cd IST_ML_PROJECT1/
  $ jupyter-notebook implementations.ipynb
  ```
  or
  ```bash
  $ jupyter-notebook run.ipynb
  ```

4. Follow the steps on each cell to produce and display results in nice HTML format.

## Team - IST

- Cheng-Chun Lee ([@wlo2398219](https://github.com/wlo2398219)) : (cheng-chun.lee@epfl.ch)
- Haziq Razali ([@haziqrazali](https://github.com/haziqrazali)): (muhammad.binrazali@epfl.ch)
- Sanadhi Sutandi ([@sanadhis](https://github.com/sanadhis)) : (i.sutandi@epfl.ch)