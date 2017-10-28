# Project 1 of Pattern Classification and Machine Learning - 2017/2018

[CS-433 PCML](http://isa.epfl.ch/imoniteur_ISAP/!itffichecours.htm?ww_i_matiere=2217650315&ww_x_anneeAcad=2017-2018&ww_i_section=249847&ww_i_niveau=&ww_c_langue=en) - [EPFL](http://epfl.ch)

> Discovery of Higgs Boson using CERN's public dataset. The submission is made to competition platform [kaggle](https://www.kaggle.com/c/epfml-higgs).

## Brief Overview
The main scripts for this project are implementations.py (six ML methods) and run.py. 
<br />In addition, we create a python notebook, namely implementations.ipynb, to nicely document and display step-by-step processes of using functions in implementations.py script. We also document how we process the features in dataset and perform our classification methods.
<br /><br />
**Note that all functions and helpers for the implementations.py and run.py are stored in scripts/ directory, for more details you can go through README in scripts/**.

## More on Technical Overview
### Data preparation and Features Removal
We devide the input data into 8 subsets based on the value of PRI_JET_NUM (feature 22) and outliers in DER_MASS_MMC (feature 1). We find out that several features are tightly coupled with the value of PRI_JET_NUM. Since PRI_JET_NUM is ranged inclusively from value of 0 until 3, we devide the input data into 4 subgroups of data based on PRI_JET_NUM value. After splitting the data, we remove features based on their strong correlation with the value of PRI_JET_NUM. The details are as follow:
1. For PRI_JET_NUM = 0, remove features: [4, 5, 6, 11, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29].
2. For PRI_JET_NUM = 1, remove features: [4, 5, 6, 11, 12, 15, 18, 20, 22, 26, 27, 28].
3. For PRI_JET_NUM = 2, remove features: [11, 15, 18, 20, 22, 28].
4. For PRI_JET_NUM = 3, remove features: [11, 15, 18, 20, 22, 28].
####
From these 4 subgroups, we devide again each subgroup into two subsets based on outliers on DER_MASS_MMC.
So at the end, we have **8 subsets** of data to obtain a model each.
<br /><br />We define this step directly in both *implementations.ipynb* and *run.py*. In run.py, it is described on **create_subsets()** and **remove_features()** functions.

### Features Processing and Generation
For each subset of input x, we process the features based on Standard score (z-score) and then expand them using logarithmic basis function and polynomial basis function. 
<br /><br />The implementations of this step are written in **preprocess.py** on scripts/preprocess.py

### Cross-validation
We validate our models using cross-validation to avoid underfitting or overfitting. Therefore we have two scripts; implementations_cross_validation.py (python script) and implementations_cross_validation.ipynb (python notebook), to show and prove that we do not encounter underfitting or overfitting in our models. Both scripts are duplication of implementations.py and implementations.ipynb except that these scripts only return the accuracies of the same methods but with cross-validation (splitting data into test and train set). 
<br /><br />For ease-to-use, please check **implementations_cross_validation.ipynb**.

## Important Notes for the Datasets
**Please simply put the two data-sets (train.csv and test.csv) in higgs-data/ directory.**

## Project Structure
- `higgs-data`: the CERN's public Higgs-Boson discovery datasets.
- `report`: report in LaTeX
- `scripts`: all main ML functions and helpers.

## implementations.py - 6 Mandatory ML Methods  
We implement 6 ML methods as follows:
1. least_squares_GD : (Linear Regression using Gradient Descent)
2. least_squares_SGD : (Linear Regression using Stochastic Gradient Descent)
3. least_squares : (Linear Regression using Normal Equations.)
4. ridge_regression : (Ridge Regression using Normal Equations)
5. logistic_regression : (Logistic Regression using Gradient Descent)
6. reg_logistic_regression : (Regularized Logistic Regression using Gradient Descent)

## run.py - Creating Final Submission File  
Our final result can be produced by executing script run.py.

* Public leaderboard
  - **82.254%** of accuracy.
* Private Leadeboard
  - **82.254%** of accuracy.

## How to use implementations.py

1. Ensure that you have python 3 in your machine.
2. Clone this repository
3. To reuse either one of six implementations of ML methods. Load the functions from implementations.py script and pass the required parameters:

  ```python
  from implementations.py import [function]
  # Example to run linear regression using gradient descent
  weights, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
  ```

  or import all methods all at once.
  ```python
  from implementations.py import *
  # Example to run least squares with normal equations
  weights, loss = least_squares(y, tx)
  ```

## How to use run.py

1. Ensure that you have python 3 in your machine.
2. Clone this repository
3. Run the run.py to create our final submission file to [kaggle](https://www.kaggle.com/c/epfml-higgs/leaderboard):

  ```bash
  $ cd IST_ML_PROJECT1/
  $ python run.py
  ```

## How to use implementations.ipynb - Jupyter Notebook

1. Install [Anaconda](https://www.continuum.io/downloads)
2. Clone this repository
3. Run the jupyter-notebook (default is enabled by conda) in terminal:

  ```bash
  $ cd IST_ML_PROJECT1/
  $ jupyter-notebook implementations.ipynb
  ```

4. Follow the steps on each cell to produce and display results in nice HTML format.

## Team - IST
[Project Repository Page](https://github.com/sanadhis/IST_ML_Project1)
- Cheng-Chun Lee ([@wlo2398219](https://github.com/wlo2398219)) : (cheng-chun.lee@epfl.ch)
- Haziq Razali ([@haziqrazali](https://github.com/haziqrazali)): (muhammad.binrazali@epfl.ch)
- Sanadhi Sutandi ([@sanadhis](https://github.com/sanadhis)) : (i.sutandi@epfl.ch)