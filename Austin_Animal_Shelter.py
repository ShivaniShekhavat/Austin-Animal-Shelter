# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:07:27 2018

@author: shekhawat
"""

%matplotlib inline
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from urllib.error import HTTPError
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.externals import joblib
import warnings
warnings.simplefilter('ignore')

#import data
endpoint = 'https://data.austintexas.gov/resource/9t4d-g238.json'
count = 1000
pages = 100


results = []
params = {'$limit': count, '$offset': 0}

for i in range(0, pages):
    
    try:
        r = requests.get(endpoint, params=params)
        results.append(r.json())
        params['$offset'] = count
        count += 1000
        
    except HTTPError as err:
        if err.response.status_code == '404':
            break
        else:
            print(err.response.status_code)

len(results)

# normalization of data
outcome_df = pd.DataFrame()

for i in results:
    outcome_df = outcome_df.append(pd.io.json.json_normalize(i))

len(outcome_df)

cats_df = outcome_df[outcome_df['animal_type'] == 'Cat'].reset_index()
del cats_df['index']

# filtered dataset
print(len(cats_df))
cats_df.head()

outcome_df.to_csv('aac_shelter_outcome.csv',index=False, encoding='utf-8')

# count of observation in dataset
cats_df['count'] = 1
cats_df['sex'] = np.where(cats_df['sex_upon_outcome'].str.contains('Male'), 'Male', 'Female')
cats_df['Spay/Neuter'] = np.where(cats_df['sex_upon_outcome'].str.contains('Intact'), 'No', 'Yes')

# heads(first5 row) of dataset we are dealing with
cats_df['age_upon_outcome'].head()

cats_df = cats_df[cats_df['age_upon_outcome'] != 'NULL']
cats_df['Periods'], cats_df['Period Range'] = cats_df['age_upon_outcome'].str.split(' ').str[0].fillna(0).astype(int), cats_df['age_upon_outcome'].str.split(' ').str[1].fillna(0)

cats_df['Period Range'].unique()


cats_df['Period Range'] = np.where(cats_df['Period Range'].str.contains('day'), 1, 
                                   np.where(cats_df['Period Range'].str.contains('week'), 7, 
                                            np.where(cats_df['Period Range'].str.contains('month'), 30, 
                                                     np.where(cats_df['Period Range'].str.contains('year'), 365, 0)))).astype(int)

cats_df['outcome_age_(days)'] = cats_df['Period Range'] * cats_df['Periods']
cats_df['outcome_age_(years)'] = cats_df['outcome_age_(days)'] / 365

#coversion of column for pandas
cats_df['date_of_birth'] = pd.to_datetime(cats_df['date_of_birth'])

cats_df['dob_year'] = cats_df['date_of_birth'].dt.year
cats_df['dob_month'] = cats_df['date_of_birth'].dt.month
cats_df['dob_monthyear'] = pd.to_datetime(cats_df['monthyear']).dt.to_period('M')

cats_df['datetime'] = pd.to_datetime(cats_df['datetime'])
cats_df['outcome_month'] = cats_df['datetime'].dt.month
cats_df['outcome_year'] = cats_df['datetime'].dt.year
cats_df['outcome_weekday'] = cats_df['datetime'].dt.weekday_name
cats_df['outcome_hour'] = cats_df['datetime'].dt.hour

cats_df['breed'].value_counts()


## visualization of data
##import data
cats_df = pd.read_csv('aac_shelter_cat_outcome_eng.csv')
sns.set(font_scale=1.7, palette=sns.color_palette("Set1", n_colors=15, desat=.5)) #sns.xkcd_palette(sns.xkcd_rgb))
sns.set_style('ticks')

p = sns.factorplot(x='outcome_type', hue='Cat/Kitten (outcome)', col='sex_upon_outcome', 
               col_wrap=3, data=cats_df, kind='count', size=4.5)

p.set_xticklabels(rotation=90)
p.set_xlabels('Outcome')
p.savefig('Outcome.png')

#factorplot 
transfer_subtype_counts = cats_df[cats_df['outcome_type'] == 'Transfer']
transfer_subtype_counts = transfer_subtype_counts.groupby(['sex_upon_outcome', 'Cat/Kitten (outcome)'])['outcome_subtype'].value_counts().reset_index(name='count')

g = sns.factorplot(x='outcome_subtype', y='count', hue='sex_upon_outcome', col='Cat/Kitten (outcome)', 
               data=transfer_subtype_counts, size=6)

g.set_xticklabels(rotation=90)
g.savefig('sex_upon_outcome.png')

## swarmplot  on basis of outcome_type
cat_outcomes_sample = cats_df.sample(frac=1.0).groupby(['outcome_type']).head(100)

plt.figure(figsize=(26,10))
g = sns.swarmplot(x='outcome_type', y='outcome_age_(years)', hue='sex_upon_outcome', 
                  data=cat_outcomes_sample, size=12)
g.tick_params(labelsize=20)
fig = g.get_figure()
fig.savefig('outcome_type.png')

## swarmplot on the basis of outcome_subtype
cat_outcomes_sample = cats_df.sample(frac=1.0).groupby(['outcome_subtype']).head(100)

plt.figure(figsize=(26,10))
g = sns.swarmplot(x='outcome_type', y='outcome_age_(years)', hue='outcome_subtype', 
                  data=cat_outcomes_sample, size=12)

g.tick_params(labelsize=20)
fig = g.get_figure()
fig.savefig('outcome_subtype.png')

## investigation to see ways to reduce no. of deaths
monthyear_outcomes = pd.pivot_table(cats_df, index=['dob_monthyear'], 
                                    columns=['outcome_type'], values='count', aggfunc=np.sum)
monthyear_outcomes.index = pd.to_datetime(monthyear_outcomes.index)

j1 = monthyear_outcomes.plot(fontsize=14, linewidth=5, title='Outcomes over Time', figsize=(26,10))
fig = j1.get_figure()
fig.savefig('outcomes over Time.png')

##Filtering of data to exclude adoption and transfers
monthyear_outcomes2 = monthyear_outcomes.filter(items=['Died', 'Disposal', 'Euthanasia', 'Missing', 
                                                      'Return to Owner', 'Rto-Adopt'])

j2 = monthyear_outcomes2.plot(fontsize=14, linewidth=5, figsize=(26,10), 
                         title='Non-Adoption/Transfer Outcomes over Time')

fig = j2.get_figure()
fig.savefig('Transfer Outcomes over Time.png')


##Significant trend on the basis of agegroup
agegroup_counts = cats_df.groupby(['age_group', 'sex'])['outcome_type'].value_counts().reset_index(name='count')

g = sns.factorplot(x='outcome_type', y='count', hue='age_group', col='sex',
                   data=agegroup_counts, size=6)

g.set_xticklabels(rotation=90)
g.savefig('agegroup outcome_type.png')

##Filtering 2yrs agegroup
agegroup_counts2 = agegroup_counts[agegroup_counts['age_group'] != agegroup_counts['age_group'].unique()[0]]

g = sns.factorplot(x='outcome_type', y='count', hue='age_group', col='sex', 
                   data=agegroup_counts2, size=6)

g.set_xticklabels(rotation=90)
g.savefig('agegroup2 outcome_type.png')


## prediction usng scikit Learn and Machine Learning
# Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):     ##This function prints and plots the confusion matrix.Normalization can be applied by setting `normalize=True`.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

## Data Prepration
cat_outcomes = pd.read_csv('aac_shelter_cat_outcome_eng.csv')   
cat_outcomes = cat_outcomes.loc[(cat_outcomes['outcome_type'] == 'Adoption') | 
                                 (cat_outcomes['outcome_type'] == 'Transfer')]
cat_outcomes.head()

## new outcome 
cat_outcomes['outcome_subtype'] = np.where(pd.isnull(cat_outcomes['outcome_subtype']), 
                                           cat_outcomes['outcome_type'], 
                                           cat_outcomes['outcome_subtype'])

# count of newly created outcome
cat_outcomes['outcome_subtype'].value_counts()

## Filteration of Data using Filter function
x = cat_outcomes.filter(items=['sex_upon_outcome', 'breed', 'color', 'coat_pattern', 
                               'domestic_breed', 'dob_month', 'age_group', 'outcome_month', 
                               'outcome_weekday', 'outcome_hour', 'Cat/Kitten (outcome)'])

## conversion of variabbles into categorical
for col in x:
    x[col] = x[col].astype('category')
    
y = cat_outcomes['outcome_subtype'].astype('category')

## Dummies creation to use categorical variables into model
xd = pd.get_dummies(x)
xnames = xd.columns
xarr = np.array(xd)

## Numerical representation using fatorize function
yarr, ynames = pd.factorize(cat_outcomes['outcome_subtype'])
x_train, x_test, y_train, y_test = train_test_split(xarr, yarr, 
                                                    test_size=0.3, random_state=1)

## Scaling of data using standardscale function
scaler = preprocessing.StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

## Random Forest
rf_classifier = RandomForestClassifier(n_estimators=250, criterion='gini', 
                                       class_weight='balanced_subsample', bootstrap=True, oob_score=True)

rf_classifier.fit(x_train, y_train)

## Important Feature
feature_importances = rf_classifier.feature_importances_

importances = pd.DataFrame({'importance': feature_importances, 'feature': xnames}, 
                           index=xnames).sort_values(by='importance', ascending=False)

sns.set(font_scale=1.50)
plt.figure(figsize=(15, 5))
plt.xticks(rotation=90)
sns.barplot(x='feature', y='importance', data=importances[0:15])


## Dimension Reduction with PCA
pca = PCA()
pca.fit(x_train_scaled)
print(len(pca.components_))

## Variance visualization
comps = np.arange(50, 250, 10)
exp_var = []

for comp in comps:
    exp_var.append(sum(pca.explained_variance_ratio_[0:comp]))
    
pca_exp_var = pd.concat([pd.Series(comps, name='components'), 
                         pd.Series(exp_var, name='explained variance')], axis=1)

pca_exp_var.set_index('components', inplace=True)

pca_exp_var.plot(kind='line', linewidth=5, figsize=(15, 5))


## Adoption and Transfer Prediction Model
feature_union = FeatureUnion([
    ('pca', PCA()),
    ('kbest', SelectKBest())
])

prediction_model = Pipeline([
    ('feature_union', feature_union),
    ('rf', RandomForestClassifier(n_estimators=1000, criterion='gini', 
                                  class_weight='balanced_subsample', bootstrap=True, 
                                  oob_score=True))
])
prediction_model.fit(x_train_scaled, y_train)

## Prediction Model Accuracy and cross- validation
cv = StratifiedKFold(2)

cv_score = cross_val_score(prediction_model, x_train_scaled, y_train, cv=cv, scoring='accuracy')
cv_predict_score = cross_val_predict(prediction_model, x_train_scaled, y_train, cv=cv)

print(cv_score)
print(np.mean(cv_score))

print(accuracy_score(y_train, cv_predict_score))

## Accuracy using predict()
rf_pred = prediction_model.predict(x_test_scaled)
print(accuracy_score(y_test, rf_pred))

## Visualization of model using Confusion Matrix
sns.set(font_scale=1.2)

rf_cnf_mat = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(7, 7))
plot_confusion_matrix(rf_cnf_mat, classes=ynames, normalize=True, title='Normalized confusion matrix')

plt.figure(figsize=(7, 7))
plot_confusion_matrix(rf_cnf_mat, classes=ynames, normalize=False, title='Non-Normalized confusion matrix')

plt.show()

print(classification_report(y_test, rf_pred, target_names=ynames))

## Finding an Optimal Model
criterion = ['gini', 'entropy']
components = [175, 200, 225]
estimators = [100, 250, 500]
k = [10, 15, 20]

feature_union = FeatureUnion([
    ('pca', PCA()),
    ('kbest', SelectKBest())
])

prediction_model_tune = Pipeline([
    ('feature_union', feature_union),
    ('rf', OneVsRestClassifier(RandomForestClassifier(class_weight='balanced_subsample', 
                                                      bootstrap=True, oob_score=True)))
])

rf_gridcv = GridSearchCV(prediction_model_tune, dict(feature_union__pca__n_components=components, 
                                                     feature_union__kbest__k=k, 
                                                     rf__estimator__n_estimators=estimators, 
                                                     rf__estimator__criterion=criterion), 
                         scoring='accuracy', cv=cv, return_train_score=True)
    
    
rf_gridcv.fit(x_train_scaled, y_train)

## conversion of result
results = pd.DataFrame(rf_gridcv.cv_results_)
results.sort_values(by='rank_test_score', ascending=False, inplace=True)
results.head()    