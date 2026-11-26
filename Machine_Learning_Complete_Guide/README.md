# Machine Learning Complete Guide

This guide takes you through 14 supervised machine learning projects using Python. Each project has its own synthetic dataset with 1,500 observations and an executed Jupyter notebook. You’ll work through the algorithm, calculate a prediction by hand, fit a pipeline, evaluate holdout results and practise explaining your choices in an interview.

**Classification** predicts a discrete label, such as Spam or Ham. Use the confusion matrix to see which classes the model mixes up and precision, recall, F1 and ROC-AUC to put numbers on those trade-offs. **Regression** predicts a continuous quantity, such as delivery time in minutes. Use MAE and RMSE to describe the size of prediction errors in that quantity’s units, then inspect residuals for patterns.

All data are synthetic and intended for education. Their predictions are unsuitable for real financial, medical, safety, educational or employment decisions.

## Projects

| # | Notebook | Dataset and task | Problem type |
| --- | --- | --- | --- |
| 01 | [Multiple Linear Regression](notebooks/01_multiple_linear_regression.ipynb) | [House Price Prediction](datasets/01_house_prices.csv) | Regression |
| 02 | [Logistic Regression](notebooks/02_logistic_regression.ipynb) | [Student Placement Prediction](datasets/02_student_placement.csv) | Classification |
| 03 | [Polynomial Regression](notebooks/03_polynomial_regression.ipynb) | [Vehicle Braking Distance Prediction](datasets/03_vehicle_braking.csv) | Regression |
| 04 | [Decision Tree Classifier](notebooks/04_decision_tree_classifier.ipynb) | [Bank Loan Approval](datasets/04_loan_approval.csv) | Classification |
| 05 | [Decision Tree Regressor](notebooks/05_decision_tree_regressor.ipynb) | [Used Car Price Prediction](datasets/05_used_car_prices.csv) | Regression |
| 06 | [Random Forest Classifier](notebooks/06_random_forest_classifier.ipynb) | [Customer Churn Prediction](datasets/06_customer_churn.csv) | Classification |
| 07 | [Random Forest Regressor](notebooks/07_random_forest_regressor.ipynb) | [Agriculture Crop Yield Prediction](datasets/07_crop_yield.csv) | Regression |
| 08 | [KNN Classifier](notebooks/08_knn_classifier.ipynb) | [Fruit Classification](datasets/08_fruit_classification.csv) | Classification |
| 09 | [KNN Regressor](notebooks/09_knn_regressor.ipynb) | [Food Delivery Time Prediction](datasets/09_food_delivery.csv) | Regression |
| 10 | [SVM Classifier](notebooks/10_svm_classifier.ipynb) | [Student Exam Result Prediction](datasets/10_exam_results.csv) | Classification |
| 11 | [SVM Regressor](notebooks/11_svm_regressor.ipynb) | [Concrete Compressive Strength Prediction](datasets/11_concrete_strength.csv) | Regression |
| 12 | [Naive Bayes Classifier](notebooks/12_naive_bayes_classifier.ipynb) | [SMS Spam Detection](datasets/12_sms_spam.csv) | Classification |
| 13 | [XGBoost Classifier](notebooks/13_xgboost_classifier.ipynb) | [Employee Attrition Prediction](datasets/13_employee_attrition.csv) | Classification |
| 14 | [XGBoost Regressor](notebooks/14_xgboost_regressor.ipynb) | [Electricity Demand Prediction](datasets/14_electricity_demand.csv) | Regression |

The [data dictionary](datasets/DATA_DICTIONARY.md) lists every column, its meaning and units, and the process used to generate each target.

## Run in Jupyter Notebook

Extract the ZIP, open a terminal and enter the extracted `Machine_Learning_Complete_Guide` directory. Python 3.12 was used for the verified runs.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

### macOS or Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m notebook
```

Open a notebook in `notebooks/` and use **Run → Run All Cells**. Outputs are already saved so you can also read the guide before installing anything. Regenerating CSV files is optional. Run `generate_datasets.py` with the same virtual environment’s Python to reproduce the bundled data. This overwrites the 14 generated CSV files in `datasets/`.

## Run in Google Colab

1. Open Colab and choose **File → Upload notebook**. Select any of the extracted `.ipynb` files.
2. Connect to the runtime. Open the **Files** sidebar and upload that notebook’s matching CSV to the runtime’s working directory, usually `/content`. Keep the original filename. The project table above gives each pair.
3. Run **Runtime → Run all**. The CSV lookup accepts a CSV in the working directory. You can also upload and extract the full ZIP inside `/content` to make all CSVs available.
4. If a required library is missing or too old, run the following in a new code cell before the project cells, restart the runtime if prompted, then run the notebook again.

```python
%pip install "numpy>=1.26,<3" "pandas>=2.2,<4" "matplotlib>=3.8,<4" "seaborn>=0.13.2,<0.14" "scikit-learn>=1.6,<2" "xgboost>=2.1,<4"
```

Upload the CSV again if Colab replaces the runtime. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Libraries and layout

NumPy generates data and handles arrays. pandas loads and summarizes the CSVs. Matplotlib and Seaborn draw the figures. scikit-learn provides preprocessing, validation, metrics and most models. XGBoost provides the two boosting estimators. Jupyter Notebook and IPython run and display the lessons.

```text
Machine_Learning_Complete_Guide/
├── README.md
├── requirements.txt
├── generate_datasets.py
├── datasets/
│   ├── DATA_DICTIONARY.md
│   └── 14 project CSVs
└── notebooks/
    └── 14 project notebooks with executed outputs
```

## Suggested learning order

Open [Multiple Linear Regression](notebooks/01_multiple_linear_regression.ipynb) first. Run one cell at a time and look at its output. Start by finding `X` (the inputs) and `y` (the answer to predict). Then follow the cells that split the rows, create the model, fit it, make predictions and score them. Sections 10?13 cover tuning, new inputs and interview practice.

The lessons use direct statements and calls to pandas, NumPy and scikit-learn. You can follow the code without knowing how to write loops, comprehensions or your own functions.

1. Start with **01–03**. Explain a regression coefficient, turn log-odds into a class probability and compare a straight line with a quadratic.
2. Work through **04–07**. Follow one decision path, then see how averaging many randomized trees changes the prediction.
3. Try **08–11**. Compare how scaling affects nearest neighbours and kernel methods, then connect the margin and epsilon tube to their losses.
4. Study **12** on its own. Calculate a spam posterior by hand before running the token-count pipeline.
5. Finish with **13–14**. Compare sequential error correction with the independently grown trees of a random forest.

For each project, calculate its worked example by hand, inspect the holdout score against its baseline, then explain one observed error or plot in your own words. Use the notebook’s project explanation to practise reporting the actual numbers.

## Algorithm comparison

| Algorithm | Learning type | Problem type | Main working principle | Important parameters | Advantages | Disadvantages | Common applications |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multiple Linear Regression | Supervised | Regression | Fits an additive conditional mean by minimizing squared errors | fit_intercept, Ridge alpha | Interpretable slopes and fast fitting | Sensitive to outliers and misspecified relationships | Housing valuation and continuous cost estimation |
| Logistic Regression | Supervised | Classification | Applies a sigmoid to an additive log-odds model | C, solver, threshold | Interpretable and probabilistic | Additive log-odds unless features are expanded | Placement analysis, demand conversion and other binary outcomes |
| Polynomial Regression | Supervised | Regression | Fits a linear model over polynomial features | degree, Ridge alpha | Captures smooth curvature with a small model | Degree choice and extrapolation require care | Curved calibration relationships and response experiments |
| Decision Tree Classifier | Supervised | Classification | Recursively splits to create purer leaves | max_depth, criterion, min_samples_split | Inspectable paths and nonlinear interactions | Unstable splits, especially for deep trees | Rule discovery and interpretable binary classification |
| Decision Tree Regressor | Supervised | Regression | Partitions inputs and predicts a leaf mean or median | max_depth, criterion, min_samples_leaf | Transparent nonlinear price segments | Stepwise estimates and high variance | Resale pricing and revenue segmentation |
| Random Forest Classifier | Supervised | Classification | Averages class probabilities from randomized trees | n_estimators, max_depth, max_features | Stable nonlinear tabular baseline | Harder to audit and costlier than a single tree | Churn, fault detection and nonlinear classification |
| Random Forest Regressor | Supervised | Regression | Averages numeric predictions from randomized trees | n_estimators, max_depth, max_features | Flexible and stable within observed ranges | Limited extrapolation and less interpretable | Predicting field yield and nonlinear tabular quantities |
| KNN Classifier | Supervised | Classification | Votes among nearby labelled examples | n_neighbors, weights, metric | Simple and supports local explanations | Sensitive to feature scales and large search spaces | Small measurement-based classification and similarity lookup |
| KNN Regressor | Supervised | Regression | Averages continuous outcomes in a neighbourhood | n_neighbors, weights, metric | Intuitive local nonlinear estimates | Sparse neighbourhoods and inference costs | Local duration, price and response estimates |
| SVM Classifier | Supervised | Classification | Learns a maximum-margin boundary, optionally through a kernel | C, kernel, gamma | Flexible decision boundary and margin scores | Needs scaling and can be costly at large sample sizes | Moderate-scale classification and pattern recognition |
| SVM Regressor | Supervised | Regression | Fits a regularized function with an epsilon-insensitive loss | C, kernel, gamma, epsilon | Smooth nonlinear fit with configurable tolerance | Sensitive to scaling and kernel choices | Calibrating continuous responses and estimating experimental properties |
| Naive Bayes Classifier | Supervised | Classification | Combines class priors with smoothed token likelihoods | alpha, ngram_range | Fast and suited to sparse counts | Ignores much of syntax and token dependence | SMS spam, email filtering and basic document classification |
| XGBoost Classifier | Supervised | Classification | Adds regularized trees to improve binary log loss | n_estimators, learning_rate, max_depth, reg_lambda | Captures nonlinear effects and interactions | Requires tuning and more effort to interpret | Tabular event likelihoods and operational classification |
| XGBoost Regressor | Supervised | Regression | Sequentially adds regularized trees to reduce numeric error | n_estimators, learning_rate, max_depth, reg_lambda | Learns interactions and flexible demand profiles | Tuning cost and weak extrapolation | Load estimation and nonlinear tabular regression |

## Which algorithms fit which problems?

- **Linear relationships** · Use multiple linear regression for an additive mean or logistic regression for additive log-odds. Both make good baselines when coefficient interpretation matters.
- **Nonlinear relationships** · Try a low-degree polynomial for smooth curvature in a few variables. Trees, forests and boosting handle thresholds and interactions. RBF SVC and SVR handle smooth nonlinear surfaces when inputs can be scaled meaningfully.
- **Text classification** · Start with CountVectorizer and Multinomial Naive Bayes for short texts and sparse counts. Keep vocabulary fitting inside training and test against unfamiliar messages.
- **Distance-based prediction** · KNN is useful when nearby examples are meaningful and it is affordable to search them. Standardize comparable measurements and check how sensitive results are to the distance metric and number of neighbours.
- **Ensemble learning** · Random forests average diverse trees to stabilize predictions. XGBoost grows trees sequentially to improve a regularized objective. Compare either with a simple baseline and tune on training folds.

