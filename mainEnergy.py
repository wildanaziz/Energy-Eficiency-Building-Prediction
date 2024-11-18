# %% [markdown]
# # Energy Eficiency Building - Wildan Aziz Hidayat

# %% [markdown]
# #### Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# %% [markdown]
# #### Data Loading

# %%
path = "/home/wildanaziz/Energy-Eficiency-Building-Prediction/dataset/ENB2012_data.xlsx"
energy = pd.read_excel(path)
energy

# %% [markdown]
# #### Change Variable Names
# 
# To read data more easily

# %%
energy.rename(columns= {'X1':'Relative_Compactness',
                        'X2':'Surface_Area',
                        'X3':'Wall_Area',
                        'X4':'Roof_Area',
                        'X5':'Overall_Height',
                        'X6':'Orientation',
                        'X7':'Glazing_Area',
                        'X8':'Glazing_Area_Distribution',
                        'Y1': 'Heating Load',
                        'Y2': 'Cooling Load'}, inplace=True)

# %%
energy.info()

# %% [markdown]
# #### Describe Datasets

# %%
energy.describe()

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# #### Checking Missing Value

# %%
check_0_variable = (energy == 0).sum()

check_0_variable

# %% [markdown]
# #### Remove Missing Value

# %%
energy = energy.loc[(energy[['Glazing_Area', 'Glazing_Area_Distribution']] != 0).all(axis=1)]

check_0_variable = (energy == 0).sum()

check_0_variable

# %% [markdown]
# #### Boxplot Visualization

# %%
sns.boxplot(x=energy['Relative_Compactness'])

# %%
sns.boxplot(x=energy['Surface_Area'])

# %%
sns.boxplot(x=energy['Wall_Area'])

# %%
sns.boxplot(x=energy['Roof_Area'])

# %%
sns.boxplot(x=energy['Overall_Height'])

# %%
sns.boxplot(x=energy['Orientation'])

# %%
sns.boxplot(x=energy['Glazing_Area'])

# %%
sns.boxplot(x=energy['Glazing_Area_Distribution'])

# %% [markdown]
# #### Handle Outliers

# %%
print(energy.dtypes)

numeric_energy = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution', 'Heating Load', 'Cooling Load']

energy[numeric_energy]

# %%
Q1 = energy[numeric_energy].quantile(0.25)
Q3 = energy[numeric_energy].quantile(0.75)
IQR = Q3 - Q1
energy=energy[~((energy[numeric_energy] < (Q1 - 1.5 * IQR)) |(energy[numeric_energy] > (Q3 + 1.5 * IQR))).any(axis=1)]

energy.shape

# %%
energy

# %% [markdown]
# ### Exploratory Data Analysis Univariate Analysis

# %%
energy[numeric_energy].hist(bins=50, figsize=(20,15))
plt.show()

# %% [markdown]
# ### Exploratory Data Analysis Mutivariate Analysis

# %%
sns.pairplot(energy, diag_kind='kde')

# %%
plt.figure(figsize=(20,15))

correlation_matrix = energy.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation Matrix of Energy Dataset")

# %% [markdown]
# ### Split into train and test

# %%
X = energy.drop(columns=['Heating Load', 'Cooling Load'], axis=1)
y = energy[['Heating Load', 'Cooling Load']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"Total sample in datasets: {len(energy)}")
print(f"Total sample in train dataset: {len(X_train)}")
print(f"Total sample in test dataset: {len(X_test)}")

# %% [markdown]
# ### Standardization

# %%
scaler = StandardScaler()

scaler.fit(X_train)

index, columns = X_train.index , X_train.columns

X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, index=index, columns=columns)
X_train.head()

# %%
X_train.describe().round(4)

# %% [markdown]
# ### Modelling
# - with KNN
# - with Random Forest Regressor 
# - with XGBoostRegressor 

# %%
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

# %% [markdown]
# #### KNN

# %%
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

result_knn = root_mean_squared_error(y_pred=knn.predict(X_train), y_true=y_train)
result_knn

# %% [markdown]
# #### Looking for best parameter tuning RF

# %%
param_grid_rf = {
    'n_estimators': [10, 15, 20, 50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_regressor = RandomForestRegressor()

rf_search = RandomizedSearchCV(
    estimator=rf_regressor,
    param_distributions=param_grid_rf,
    n_iter=10,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=123
)

rf_search.fit(X_train, y_train)

print("Best parameters for Random Forest Regressor:\n", rf_search.best_params_)

# %% [markdown]
# #### Combine best parameter tuning with my tuning

# %%
rfRegressor = RandomForestRegressor(n_estimators=1000, max_depth=6, min_samples_split=4, min_samples_leaf=4, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=123)
rfRegressor.fit(X_train, y_train)
result_rf = root_mean_squared_error(y_pred=rfRegressor.predict(X_train), y_true=y_train)
result_rf

# %% [markdown]
# #### Looking for best parameter tuning XGB

# %%
param_grid_xgb = {
    'n_estimators': [300, 400, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [1, 5, 10, 20],
    'reg_lambda': [1, 5, 10, 50],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

xgb = XGBRegressor(objective="reg:squarederror", random_state=123)

xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_iter=50,
    verbose=1,
    n_jobs=-1,
    random_state=123
)

xgb_search.fit(X_train, y_train)

best_params = xgb_search.best_params_
print("Best parameters for XGBRegressor:", best_params)

# %% [markdown]
# #### Combine best parameter tuning with my tuning

# %%
xgb = XGBRegressor(n_estimators=1000, eval_metric='rmse', max_depth=10, min_child_weight=1, learning_rate=0.01, reg_alpha=10, reg_lambda=10, subsample=0.7, colsample_bytree=1.0, random_state=123, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
result_xgb = root_mean_squared_error(y_pred=xgb.predict(X_train), y_true=y_train)
result_xgb

# %% [markdown]
# ### Model Evaluation

# %%
test_index, test_columns = X_test.index , X_test.columns

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, index=test_index, columns=test_columns)
X_test.head()

# %% [markdown]
# #### Looking for train and test

# %%
all_model_result = pd.DataFrame(columns=['train_rmse', 'test_rmse'], index=['knn', 'rf', 'xgbr'])

model_dictionary = {
    'knn': knn,
    'rf': rfRegressor,
    'xgbr': xgb
}

for model_name, model in model_dictionary.items():
    
    all_model_result.loc[model_name]['train_rmse'] = root_mean_squared_error(y_pred=model.predict(X_train), y_true=y_train)
    all_model_result.loc[model_name]['test_rmse'] = root_mean_squared_error(y_pred=model.predict(X_test), y_true=y_test)
    
all_model_result

# %% [markdown]
# #### Plot train and test all models

# %%
ax = plt.subplot()

all_model_result.sort_values('test_rmse').plot(kind='bar', figsize=(20,10), ax=ax)

# %% [markdown]
# #### Trying predicting using all models

# %%
prediction = X_test.iloc[:10].copy()
pred_dict = {
    'Heating Load (true)': y_test['Heating Load'][:10].values,
    'Cooling Load (true)': y_test['Cooling Load'][:10].values
}

for name, model in model_dictionary.items():
    pred_heating, pred_cooling = model.predict(prediction).T  
    pred_dict[name + '_Heating Load (pred)'] = pred_heating.round(1)
    pred_dict[name + '_Cooling Load (pred)'] = pred_cooling.round(1)

pred_df = pd.DataFrame(pred_dict)
pred_df


