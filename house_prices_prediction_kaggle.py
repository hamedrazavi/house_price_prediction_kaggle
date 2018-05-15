# # House prices prediction (kaggle.com)
# In the Ames Housing dataset on kaggle, we would like to predict house prices with regression models. Thanks to @Serigne on kaggle.com for some helpful hints.

# Import Libraries
import pandas as pd
from sklearn import model_selection, linear_model, ensemble, metrics
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import skew
import lightgbm as lgb
import house_prices_functions
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points


# Import and explore the data
trData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
data_list = [trData, testData]
combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)
ltr = len(trData)
ltest = len(testData)

# Clean, fill in the missing data, and remove outliers
# What are the missing data?
def CheckNull(df, feature):
    print(df[feature].isnull().sum())

# Fill in the missing data
# We fill in the missing data with either mean() or mode().
testData['TotalBsmtSF'].fillna(0, inplace=True)  # Nan = no basement
testData['GarageArea'].fillna(0, inplace=True)  # Nan means no garage so 0 area
testData['MSZoning'].fillna(testData['MSZoning'].mode()[0], inplace=True)
testData['SaleType'].fillna(testData['SaleType'].mode()[0], inplace=True)
for data in data_list:
    data['LotFrontage'].fillna(combinedData['LotFrontage'].mean(), inplace=True)
    data['BsmtQual'].fillna('None', inplace=True)
    data['Functional'].fillna(data['Functional'].mode()[0], inplace=True)
    data['GarageYrBlt'].fillna(0, inplace=True)  # 0 to indicate not garage; GarageYrBlt will be regarded as categorial
    data['GarageCars'].fillna(0, inplace=True)  # Nan values mean no garage so 0 cars
    data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)
    data['MasVnrType'].fillna('None', inplace=True)  # Nan = none
    data['MasVnrArea'].fillna(0, inplace=True)  # nan = 0
    data['BsmtCond'].fillna('None', inplace=True)
    data['BsmtExposure'].fillna('None', inplace=True)
    data['GarageType'].fillna('None', inplace=True)
    data['BsmtFinSF1'].fillna(0, inplace=True)  # Nan = 0
    data['BsmtFinSF2'].fillna(0, inplace=True)  # Nan = 0
    data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean(), inplace=True)
    data['PoolQC'].fillna('None', inplace=True)
    data['MiscFeature'].fillna('None', inplace=True)
    data['Alley'].fillna('None', inplace=True)
    data['Fence'].fillna('None', inplace=True)
    data['BsmtFullBath'].fillna(0, inplace=True)
    data['BsmtHalfBath'].fillna(0, inplace=True)
    data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=True)
    data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0], inplace=True)
    data['BsmtFinType2'].fillna('None', inplace=True)
    data['FireplaceQu'].fillna('None', inplace=True)
    data['GarageFinish'].fillna('None', inplace=True)
    data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=True)
    data['BsmtFinType1'].fillna('None', inplace=True)
    data['GarageQual'].fillna('None', inplace=True)
    data['GarageCond'].fillna('None', inplace=True)
    data['Utilities'].fillna('None', inplace=True)

for data in data_list:
    data['HasDeckPorch'] = ((data['WoodDeckSF'] > 0) | (data['OpenPorchSF'] > 0) | (data['EnclosedPorch']) > 0 |
                            (data['3SsnPorch'] > 0) | (data['ScreenPorch'] > 0)) * 1
    data['HasPool'] = (data['PoolArea'] > 0) * 1

# The following variables are non-numeric but also can be regarded as non-categorial:
# LandContour, Utilities, LandSlope, OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond,
#  BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, Functional, FireplaceQu, GarageFinish,
# GarageQual, GarageCond, PavedDrive, PoolQC, Fence
# We expected that a Level countour be more expensive, but it does not seem to be the case on average. We decide to drop this feature as it does not seem meaningfull in relation to the saleprice.


#LandContour
combinedData = house_prices_functions.combineTrTest(trData, testData)
combinedData['LandContour'].replace(['Bnk', 'Lvl', 'Low', 'HLS'], [0, 1, 2, 3], inplace=True)
# As there seem to be no correlation (even a stragne correlation that in Severe slope the price per area is higher!) we drop this feature.
combinedData.drop('LandSlope', 1, inplace=True)
combinedData.drop('Utilities', 1, inplace = True)
# There deoes not seem to be a meaningful correlation, so we drop this feature:
combinedData.drop('MiscFeature', 1, inplace=True)
# When we graph price per area vs overal quality, to our surprise, the Overal qual = 9 has a higher price per area than 10! So, we combine, 9 and 10. Also, 1, 2, and 3, will be replace by 1.
# combinedData['OverallQual'].replace([1, 2, 3, 9, 10], [1, 1, 1, 9, 9], inplace = True)
combinedData['ExterQual'].replace(['Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3], inplace=True)
combinedData['ExterCond'].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 2, 4], inplace=True)
combinedData['BsmtQual'].replace(['None', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4], inplace=True)
combinedData['BsmtCond'].replace(['Po', 'None', 'Fa', 'TA', 'Gd'], [0, 1, 2, 3, 4], inplace=True)
combinedData['BsmtExposure'].replace(['None', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4], inplace=True)
combinedData['Functional'].replace(['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
                                   [0, 0, 1, 1, 1, 1, 2], inplace=True)
combinedData['CentralAir'].replace(['N', 'Y'], [0, 1], inplace=True)
combinedData['PavedDrive'].replace(['N', 'P', 'Y'], [0, 1, 2], inplace=True)
# There does not seem to be a meaningful correlation, also, because given that no fence has a better average price than fence with Good privacy, and not much correlation with price per area we drop this feature:
combinedData.drop('Fence', 1, inplace=True)
combinedData.drop('PoolQC', 1, inplace = True)
combinedData['HeatingQC'].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4], inplace=True)
combinedData['FireplaceQu'].replace(['Po', 'None', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5], inplace=True)
combinedData['KitchenQual'].replace(['Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3], inplace=True)
combinedData['GarageQual'].replace(['Po', 'None', 'Fa', 'TA', 'Gd', 'Ex'], [0, 0, 1, 2, 3, 4], inplace=True)
combinedData['GarageCond'].replace(['Po', 'None', 'Fa', 'TA', 'Gd', 'Ex'], [0, 0, 0, 1, 1, 1], inplace=True)
combinedData['BsmtFinType1'].replace(['None', 'LwQ', 'Rec', 'BLQ', 'Unf', 'ALQ', 'GLQ'], [0, 1, 2, 3, 4, 4, 5],
                                     inplace=True)
combinedData['BsmtFinType2'].replace(['None', 'BLQ', 'Rec', 'LwQ', 'Unf', 'ALQ', 'GLQ'], [0, 1, 2, 2, 3, 4, 4],
                                     inplace=True)
combinedData['GarageFinish'].replace(['None', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3], inplace=True)
# Similarly, for month of sale, there seem to be little correlation. So, we drop this feature as well:
combinedData.drop('MoSold', 1, inplace=True)
ltr = len(trData)
ltest = len(testData)
trData = combinedData[0:ltr].reset_index(drop=True)
testData = combinedData[ltr:ltr + ltest].reset_index(drop=True)
combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)

# Graph the price (output) and features (input) distributions:
# For simplicity we will convert the price to $1000:
trData['SalePriceK'] = trData['SalePrice'] / 1000
trData['PricePerArea'] = trData['SalePrice'] / trData['GrLivArea']

# ### 2.3 Remove the outliers
trData.drop(trData[(trData['GrLivArea'] > 4000) & (trData['SalePrice'] < 300000)].index, inplace=True)
combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)

# After removing the two outliers:
# Another possiblity for outliers is the price per area. There is only one house with a price per square foot less than $31, and the next lowest price is $40, so, we remove the cheapest one:
trData[trData['SalePrice'] / trData['GrLivArea'] < 31]
# trData.drop(trData[trData['SalePrice'] / trData['GrLivArea'] < 31].index, inplace=True)

data_list = [trData, testData]
for data in data_list:
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

# House age seems to be an important feature correlated with the saleprice:
# for data in data_list:
#     data['HouseAge'] = data['YrSold'] - data['YearBuilt']
#     data['YrfromRemod'] = data['YrSold'] - data['YearRemodAdd']
# trData.drop('YearBuilt', 1, inplace=True)
# testData.drop('YearBuilt', 1, inplace=True)


combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)
combinedData[combinedData['GarageYrBlt'] > 2010]['GarageYrBlt']
combinedData.loc[combinedData['GarageYrBlt'] > 2010, 'GarageYrBlt'] = 2010
combinedData['GarageCars'].replace([5], 4, inplace=True)
#since GarageArea very correlated with GarageCars:
combinedData.drop('GarageArea', 1, inplace = True)
combinedData['PoolArea'].value_counts()
# We drop the PoolArea as there seem to be no reliable information in it:
combinedData.drop('PoolArea', 1, inplace=True)
# We drop the feature for obvious reasons:
combinedData.drop('MiscVal', 1, inplace=True)
combinedData['TotRmsAbvGrd'].replace([13, 14, 15], 13, inplace=True)

# #### 3.2.1 Convert categorial to dummies
# The following variables are categorial type features:
#
# MSSubClass, MSZoning, Alley, LotShape, LotConfig, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation, Heating, Electrical, GarageType, SaleType, SaleCondition

ltr = len(trData)
ltest = len(testData)
combinedData.drop(['Alley', 'Street', 'LotShape', 'LotConfig', 'Electrical', 'GarageYrBlt', 'Heating'], 1, inplace=True)
#which columns are used eventually?
print(combinedData.columns)
combinedData = pd.get_dummies(combinedData, columns=['MSSubClass', 'MSZoning',
                                                     'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                                                     'MasVnrType', 'Foundation', 'GarageType',
                                                     'SaleType', 'SaleCondition', 'KitchenAbvGr'])
# trData = pd.get_dummies(trData, columns=['MSSubClass', 'MSZoning', 'Alley', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition'])
# testData = pd.get_dummies(testData, columns=['MSSubClass', 'MSZoning', 'Alley', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition'])
trData = combinedData[0:ltr].reset_index(drop=True)
testData = (combinedData[ltr:ltr + ltest].reset_index(drop=True)).drop('SalePrice', 1)
combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)
conditionList = combinedData['Condition1'].unique()
Condition1Dummies = pd.get_dummies(combinedData['Condition1'], columns=['Condition1'])
Condition2Dummies = pd.get_dummies(combinedData['Condition2'], columns=['Condition2'])
ConditionDummies = Condition1Dummies
for colName in Condition2Dummies.columns:
    ConditionDummies[colName] += Condition2Dummies[colName]
    ConditionDummies[colName] = (ConditionDummies[colName] > 0) * 1  # To prevent duplicates
ConditionDummies.drop('Norm', 1, inplace=True);
for colName in ConditionDummies.columns:
    trData[colName] = ConditionDummies[colName][0:ltr].reset_index(drop=True)
    testData[colName] = ConditionDummies[colName][ltr:ltr + ltest].reset_index(drop=True)

exteriorList = combinedData['Exterior1st'].unique()
exterior1Dummies = pd.get_dummies(combinedData['Exterior1st'], columns=['Exterior1st'])
exterior2Dummies = pd.get_dummies(combinedData['Exterior2nd'], columns=['Exterior2nd'])
exterior2Dummies.rename(columns={'Brk Cmn': 'BrkComm', 'CmentBd': 'CemntBd', 'Wd Shng': 'WdShing'}, inplace=True);
exteriorDummies = exterior2Dummies
for colName in exterior1Dummies.columns:
    exteriorDummies[colName] += exterior1Dummies[colName]
    exteriorDummies[colName] = (exteriorDummies[colName] > 0) * 1  # To prevent duplicates
for colName in exteriorDummies.columns:
    trData[colName] = exteriorDummies[colName][0:ltr].reset_index(drop=True)
    testData[colName] = exteriorDummies[colName][ltr:ltr + ltest].reset_index(drop=True)
data_list = [trData, testData]
for data in data_list:
    data.drop('Condition1', 1, inplace=True)
    data.drop('Condition2', 1, inplace=True)
    data.drop('Exterior1st', 1, inplace=True)
    data.drop('Exterior2nd', 1, inplace=True)
# #### Regularization: (thanks to @papiu on kaggle.com)
combinedData = pd.concat(objs=[trData, testData], axis=0).reset_index(drop=True)

# log transform skewed numeric features:
numeric_feats = combinedData.dtypes[combinedData.dtypes != "object"].index
skewed_feats = combinedData[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 1.25]
skewed_feats = skewed_feats.index
combinedData[skewed_feats] = np.log1p(combinedData[skewed_feats])
ltr = len(trData)
ltest = len(testData)
trData = combinedData[0:ltr].reset_index(drop=True)
testData = combinedData[ltr:ltr + ltest].reset_index(drop=True).drop(['SalePrice', 'SalePriceK', 'PricePerArea'], 1)

# 4. Train the models and predict
X = trData.drop(['SalePrice', 'SalePriceK', 'PricePerArea', 'Id'], 1)
y = trData['SalePrice']

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.5, train_size=0.5, random_state=1)
n_folds = 10


kRR = KernelRidge(alpha=2, degree=1)

score = [0, 0]
kfold = 5
for i in range(kfold):
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=i)
    score = [sum(x) for x in zip(score, house_prices_functions.find_cv_error(Xtrain, ytrain))]
score = [x / kfold for x in score]
print(score[0], " ", score[1])

#final Crossvalidation
clfList = [linear_model.LinearRegression(), ensemble.RandomForestRegressor(), ensemble.GradientBoostingRegressor(),
           xgb.XGBRegressor(), KernelRidge(), linear_model.BayesianRidge(), lgb.LGBMRegressor(verbose = -1)]
cvSplit = model_selection.ShuffleSplit(n_splits=10, train_size=0.5, test_size=0.5, random_state=0)
maxDepthList = [2, 4]
nEstimatorsList = [400, 500]
num_leavesList = [4, 5]
etaList = [0.1, 0.05, 0.01]
rndStateList = [0, 1, 2]
gammaList = [0]
colsample_bytreeList = [0.4]
alphaList = [4]
degreeList = [1]
gridBool = [True, False]
paramGridList = [
    [{'fit_intercept': gridBool}], [{'max_depth': [4, 10], 'random_state': rndStateList}],
    [{'n_estimators': nEstimatorsList, 'max_depth': maxDepthList, 'random_state': rndStateList}],
    [{'max_depth': maxDepthList, 'gamma': gammaList, 'colsample_bytree': colsample_bytreeList}],
    [{'alpha': alphaList, 'degree': degreeList}], [{}],
    [{'num_leaves': num_leavesList, 'n_estimators': nEstimatorsList}]
]
bestScoreList = []
for clf, param in zip(clfList, paramGridList):
    bestSearch = model_selection.GridSearchCV(estimator=clf, param_grid=param,
                                              cv=cvSplit, scoring='neg_mean_squared_error', n_jobs=4)
    bestSearch.fit(X, y)
    bestParam = bestSearch.best_params_
    bestScore = round((-bestSearch.best_score_) ** 0.5, 5)
    print('The best parameter for {} is {} with a runtime of seconds with an error of {}'.format(clf.__class__.__name__,
                                                                                                 bestParam, bestScore))
    clf.set_params(**bestParam)
    bestScoreList.append(bestScore)
print("--" * 45, "\nMax cross-validation score is {}".format(round(min(bestScoreList), 5)))
print("--" * 45, "\nAverage cross-validation score is {}".format(sum(sorted(bestScoreList, reverse=False)[0:2]) / 2))


# Thanks to Serigne on kaggle.com
class AveragingModels():
    def __init__(self, models, coeffs):
        self.models = models
        self.coeff = coeffs
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        pred = 0
        for i in range(len(self.models)):
            pred += self.coeff[i] * predictions[:, i]
        return pred

# averaging
bayR = linear_model.BayesianRidge()
averagingC = AveragingModels(models=(clfList[2], clfList[3], clfList[4], clfList[5], clfList[6]),
                             coeffs=[0.1, 0.1, 0.45, 0.2, 0.15])

averagingC.fit(Xtrain, ytrain)  # Note we fit the Whole X, y
arpredict = averagingC.predict(Xtest)
print(metrics.mean_squared_error(ytest, arpredict) ** 0.5)
predData = pd.DataFrame({'Index': ytest.index, 'SalePrice': ytest.values, 'SalePricePredicted': arpredict,
                         'Error': arpredict - ytest.values})

averagingC = AveragingModels(models=(clfList[2], clfList[3], clfList[4], clfList[5], clfList[6]),
                             coeffs=[0.1, 0.1, 0.35, 0.35, 0.1])
bayR = linear_model.BayesianRidge()

house_prices_functions.eval_cv(averagingC, X, y, 5)

averagingC.fit(X, y)  # Note we fit the Whole X, y
arpredict = averagingC.predict(Xtest)
print(metrics.mean_squared_error(ytest, arpredict) ** 0.5)
predData = pd.DataFrame({'Index': ytest.index, 'SalePrice': ytest.values, 'SalePricePredicted': arpredict,
                         'Error': arpredict - ytest.values})

trsh = 50000
print(len(Xtest[abs(np.expm1(arpredict) - np.expm1(ytest.values)) > trsh]))
predData[abs(arpredict - ytest.values) > trsh]
plt.plot((np.expm1(arpredict)) - np.expm1(ytest), '.')
plt.plot((np.expm1(ytest)), '.')
plt.hist(Xtest[abs(arpredict - ytest.values) > trsh]['OverallQual'])
ytest[abs(arpredict - ytest.values) > 100000]
testDataTemp = testData.drop(['Id'], 1)
arpredict = averagingC.predict(testDataTemp)
arpredict = np.expm1(arpredict)
ypredict = pd.DataFrame({'Id': testData['Id'], 'SalePrice': arpredict})
ypredict.to_csv('../predictions.csv', index=False)
ypredict.head()
yold = pd.read_csv('../predictions11644.csv')
yold.head()
plt.plot(yold)
plt.plot(ypredict)
plt.plot(np.expm1(trData['SalePrice']))
print(metrics.mean_squared_log_error(yold, ypredict) ** 0.5)


