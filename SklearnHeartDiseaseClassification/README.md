# Heart-Disease-classification-ML
## In this notebook we are trying to predict whether or not patient has heart disease. 

During this project we will look at the following topics:

* Exploratory data analysis (EDA) - the process of going through a dataset and finding out more about it.
* Model training - create model(s) to learn to predict a target variable based on other variables.
* Model evaluation - evaluating a models predictions using problem-specific evaluation metrics.
* Model comparison - comparing several different models to find the best one.
* Model fine-tuning - once we've found a good model, we are going to improve it.
* Saving and loading our model - after we have finished hyperparameters tuning we are going to save and load our model using joblib liblary.
* Feature importance - we will find features that have the most impact on target variable (whether or not patient has heart disease).
* Cross-validation - we ware going to use cross-val-score from sklearn liblary to get overwiev how good our model really is (how it generalises).
* Reporting what we've found - we are going to create several plots (using matplotlib and seaborn) to make our results user-friendly.

To work through these topics, we'll use Pandas, Matplotlib and NumPy for data anaylsis, as well as, Scikit-Learn for machine learning and modelling tasks.

## Data

The original data came from the Cleveland database from UCI Machine Learning Repository.
Howevever, we've downloaded it in a formatted way from Kaggle.
The original database contains 76 attributes, but here only 14 attributes will be used. Attributes (also called features) are the variables what we'll use to predict our target variable.
Attributes and features are also referred to as independent variables and a target variable can be referred to as a dependent variable.
We use the independent variables to predict our dependent variable.
Or in our case, the independent variables are a patients different medical attributes and the dependent variable is whether or not they have heart disease.

## Evaluation

If we can reach 85% accuracy (this metric may change overtime) at predicting whether or not a patient has heart disease during the proof of concept, we'll pursure this project.

## Features inside heart-disease dataset

* age - age in years
* sex - (1 = male; 0 = female)
* cp - chest pain type
* 0: Typical angina: chest pain related decrease blood supply to the heart
* 1: Atypical angina: chest pain not related to heart
* 2: Non-anginal pain: typically esophageal spasms (non heart related)
* 3: Asymptomatic: chest pain not showing signs of disease
* trestbps - resting blood pressure (in mm Hg on admission to the hospital)
* anything above 130-140 is typically cause for concern
* chol - serum cholestoral in mg/dl
* serum = LDL + HDL + .2 * triglycerides
* above 200 is cause for concern
* fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
* '>126' mg/dL signals diabetes
* restecg - resting electrocardiographic results
* 0: Nothing to note
* 1: ST-T Wave abnormality
* can range from mild symptoms to severe problems
* signals non-normal heart beat
* 2: Possible or definite left ventricular hypertrophy
* Enlarged heart's main pumping chamber
* thalach - maximum heart rate achieved
* exang - exercise induced angina (1 = yes; 0 = no)
* oldpeak - ST depression induced by exercise relative to rest
* looks at stress of heart during excercise
* unhealthy heart will stress more
* slope - the slope of the peak exercise ST segment
* 0: Upsloping: better heart rate with excercise (uncommon)
* 1: Flatsloping: minimal change (typical healthy heart)
* 2: Downslopins: signs of unhealthy heart
* ca - number of major vessels (0-3) colored by flourosopy
* colored vessel means the doctor can see the blood passing through
* the more blood movement the better (no clots)
* thal - thalium stress result
* 1,3: normal
* 6: fixed defect: used to be defect but ok now
* 7: reversable defect: no proper blood movement when excercising
* target - have disease or not (1=yes, 0=no) (= the predicted attribute)
* Note: No personal identifiable information (PPI) can be found in the dataset.
