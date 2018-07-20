import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import model_selection, ensemble, linear_model
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics

import pandas as pd

def PlotCorr(df, feature):
    sns.barplot(x=df[feature], y=df['SalePrice'])
    print(df[feature].value_counts())
def PlotCorr2(df, feature):
    sns.barplot(x=df[feature], y=df['PricePerArea'])
    print(df[feature].value_counts())
def combineTrTest(dfTr, dfTest):
    dfCombined = pd.concat(objs=[dfTr, dfTest], axis=0).reset_index(drop=True)
    return (dfCombined)


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

def find_cv_error(Xtrain, ytrain):
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
        bestSearch.fit(Xtrain, ytrain)
        bestParam = bestSearch.best_params_
        bestScore = round((-bestSearch.best_score_) ** 0.5, 5)
        print('The best parameter for {} is {} with a runtime of seconds with an error of {}'.format(
            clf.__class__.__name__, bestParam, bestScore))
        clf.set_params(**bestParam)
        bestScoreList.append(bestScore)
    print("--" * 45, "\nMax cross-validation score is {}".format(round(min(bestScoreList), 5)))
    print("--" * 45,
          "\nAverage cross-validation score is {}".format(sum(sorted(bestScoreList, reverse=False)[0:2]) / 2))
    return [round(min(bestScoreList), 5), sum(sorted(bestScoreList, reverse=False)[0:2]) / 2]

def eval_cv(clf, X, y, cvNum):
    score = []
    for i in range(cvNum):
        Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.5, train_size=0.5,
                                                                        random_state=i)
        clf.fit(Xtrain, ytrain)  # Note we fit the Whole X, y
        arpredict = clf.predict(Xtest)
        score.append(metrics.mean_squared_error(ytest, arpredict) ** 0.5)
    return sum(score) / len(score)

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

def rmsle_cv(model):
    kf = model_selection.KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse = np.sqrt(-model_selection.cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (sum(rmse) / n_folds)