Project 1 of Pattern Classification and Machine Learning - 2017/2018

[CS-433 PCML] - [EPFL]

Aim:
Simulate discovery of Higgs Boson using CERN's public dataset. 
The submission is made to Kaggle competition platform.

Brief Overview:
The main scripts for this project are implementations.py and run.py. 
In addition, we create python notebook, namely implementations.ipynb, to nicely document, reuse, and display 
processes in implementations.py scripts step-by-step.
Note that all functions and helpers for the implementations.py and run.py are stored in scripts/ directory.

Important Notes for the Data
Please simply put the two data-sets (train.csv and test.csv) in higgs-data/ directory.

Project Structure:
- higgs-data: the CERN's public Higgs-Boson discovery data.
- report: report in LaTeX
- scripts: all main ML functions and helpers.
- setup: Contains readme to give git syntax cheatsheet.

We implement 6 ML methods in implementations.py as follows:
1. least_squares_GD : (Linear Regression using Gradient Descent)
2. least_squares_SGD : (Linear Regression using Stochastic Gradient Descent)
3. least_squares : (Linear Regression using Normal Equations.)
4. ridge_regression : (Ridge Regression using Normal Equations)
5. logistic_regression : (Logistic Regression using Gradient Descent)
6. reg_logistic_regression : (Regularized Logistic Regression using Gradient Descent)

We produce our final submission using run.py with the result as follows:
Public leaderboard:
  - 82.254% of accuracy.
Private Leadeboard:
  - 82.254% of accuracy.


How to use implementations.py:
1. Ensure that you have python 3 in your machine.
2. To reuse either one of six implementations of ML methods, load the functions from implementations.py script and pass the required parameters:
  # In python
  from implementations.py import *
  # Example to run linear regression using gradient descent
  weights, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)


How to use run.py:
1. Ensure that you have python 3 in your machine.
2. Run the run.py to create our final submission file to kaggle:
  # Execute in shell terminal
  $ python run.py


How to use implementations.ipynb - Jupyter Notebook:
1. Install Anaconda
2. Run the jupyter-notebook (default is enabled by conda) in terminal:
  # Execute in shell terminal  
  $ jupyter-notebook implementations.ipynb
3. Follow the steps on each cell to produce and display results in nice HTML format.


Team - IST:
- Cheng-Chun Lee : (cheng-chun.lee@epfl.ch)
- Haziq Razali : (muhammad.binrazali@epfl.ch)
- Sanadhi Sutandi : (i.sutandi@epfl.ch)