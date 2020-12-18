import warnings

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

warnings.simplefilter(action='ignore', category=FutureWarning)

from optuna.samplers import CmaEsSampler
import joblib
import os.path
from os import path
import pandas as pd
import numpy as np
import optuna
import datawig
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, PowerTransformer, \
    QuantileTransformer, Normalizer, PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, make_scorer, \
    f1_score
import scipy.stats

datawig_x_train, datawig_x_test, datawig_y_train, datawig_y_test = None, None, None, None
titanic_dataframe = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)


class PreparedData:
    def __init__(self, p, r, f, c):
        self.precision = p
        self.recall = r
        self.f1score = f
        self.classifier = c


def prepare_data_drop():
    data_for_drop = titanic_dataframe.copy()

    xtrain_drop = data_for_drop.head(400)
    xtest_drop = data_for_drop.tail(100)

    xtrain_drop = xtrain_drop.dropna(axis='rows')
    xtest_drop = xtest_drop.dropna(axis='rows')

    yvalues_train_drop = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues_train_drop["Survived"] = xtrain_drop["Survived"].copy()

    yvalues_test_drop = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues_test_drop["Survived"] = xtest_drop["Survived"].copy()

    xtrain_drop.drop('Survived', axis=1, inplace=True)
    xtrain_drop.drop('PassengerId', axis=1, inplace=True)

    xtest_drop.drop('Survived', axis=1, inplace=True)
    xtest_drop.drop('PassengerId', axis=1, inplace=True)

    return xtrain_drop, xtest_drop, yvalues_train_drop, yvalues_test_drop


def prepare_data_mean():
    data = titanic_dataframe.copy()
    yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues["Survived"] = data["Survived"].copy()

    data.drop('Survived', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)

    x_train = data.head(400)
    x_train = x_train.fillna(x_train.mean())

    x_test = data.tail(100)
    x_test = x_test.fillna(x_test.mean())

    y_train = yvalues.head(400)
    y_test = yvalues.tail(100)

    return x_train, x_test, y_train, y_test


def prepare_data_median():
    data = titanic_dataframe.copy()
    yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues["Survived"] = data["Survived"].copy()

    data.drop('Survived', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)

    x_train = data.head(400)
    x_train = x_train.fillna(x_train.median())

    x_test = data.tail(100)
    x_test = x_test.fillna(x_test.median())

    y_train = yvalues.head(400)
    y_test = yvalues.tail(100)

    return x_train, x_test, y_train, y_test


def prepare_data_zero():
    data = titanic_dataframe.copy()
    yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues["Survived"] = data["Survived"].copy()

    data.drop('Survived', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)

    x_train = data.head(400)
    x_train = x_train.fillna(0.0)

    x_test = data.tail(100)
    x_test = x_test.fillna(0.0)

    y_train = yvalues.head(400)
    y_test = yvalues.tail(100)

    return x_train, x_test, y_train, y_test


def remove_outliers_zscore(threshold, xtrain, xtest, ytrain, ytest):

    df = xtrain
    df['zscore'] = (df.Age - df.Age.mean()) / df.Age.std(ddof=0)

    df2 = ytrain
    df2['zscore'] = df['zscore']

    df = df[(df.zscore <= threshold) & (df.zscore >= -threshold)].copy()
    df2 = df2[(df.zscore <= threshold) & (df2.zscore >= -threshold)].copy()

    df.drop('zscore', axis=1, inplace=True)
    df2.drop('zscore', axis=1, inplace=True)

    # print(df.head())
    # print(df.info())
    # print(df.describe())

    return df, xtest, df2, ytest


def print_classifier_stats(classifier, xtrain, xtest, ytrain, ytest):
    predictions = classifier.predict(xtest)
    matrix = confusion_matrix(ytest, predictions)

    tn, fp, fn, tp = matrix.ravel()
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)

    print(classifier)
    print(matrix)
    print(classification_report(ytest, predictions))

    print("Accuracy: " + str((tn + tp) / (tn + tp + fp + fn) * 100) + "%")
    print("Precision: " + str(precision_score(ytest, predictions)))

    print("Recall: " + str(recall_score(ytest, predictions)))
    print("F1 score: " + str(f1_score(ytest, predictions)))


def prepare_data_datawig():

    global datawig_x_train
    global datawig_x_test
    global datawig_y_train
    global datawig_y_test

    if datawig_x_train is not None:
        return datawig_x_train, datawig_x_test, datawig_y_train, datawig_y_test

    data = titanic_dataframe.copy()
    yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues["Survived"] = data["Survived"].copy()

    data.drop('Survived', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)

    x_train = data.head(400)
    x_train = datawig.SimpleImputer.complete(x_train)

    x_test = data.tail(100)
    x_test = datawig.SimpleImputer.complete(x_test)

    y_train = yvalues.head(400)
    y_test = yvalues.tail(100)

    datawig_x_train = x_train
    datawig_x_test = x_test
    datawig_y_train = y_train
    datawig_y_test = y_test

    return datawig_x_train, datawig_x_test, datawig_y_train, datawig_y_test


def scale_data_with(scaler, x_train, x_test, y_train, y_test):
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def prepare_data_by_parameter(activation):
    if activation == 'drop':
        return prepare_data_drop()
    if activation == 'mean':
        return prepare_data_mean()
    if activation == 'median':
        return prepare_data_median()
    if activation == 'zero':
        return prepare_data_zero()
    if activation == 'datawig':
        return prepare_data_datawig()


def select_scaler_by_parmeter(scaler):
    if scaler == 'MinMaxScaler':
        return MinMaxScaler()
    if scaler == 'MaxAbsScaler':
        return MaxAbsScaler()
    if scaler == 'RobustScaler':
        return RobustScaler()
    if scaler == 'StandardScaler':
        return StandardScaler()
    if scaler == 'QuantileTransformer':
        return QuantileTransformer()
    if scaler == 'PowerTransformer':
        return PowerTransformer()
    if scaler == 'PolynomialFeatures':
        return PolynomialFeatures()
    if scaler == 'Normalizer':
        return Normalizer()


def objective(trial):

    activation = trial.suggest_categorical('missing_data_fix', ['drop', 'mean', 'median', 'zero', 'datawig'])
    x_train, x_test, y_train, y_test = prepare_data_by_parameter(activation)

    #threshold = trial.suggest_uniform('threshold', 1.0, 4.0)
    #x_train, x_test, y_train, y_test = remove_outliers_zscore(threshold, x_train, x_test, y_train, y_test)

    scaler = trial.suggest_categorical('scaler',
                                       ['MinMaxScaler',
                                        'MaxAbsScaler',
                                        'RobustScaler',
                                        'StandardScaler',
                                        'QuantileTransformer',
                                        'PowerTransformer',
                                        'PolynomialFeatures'])

    x_train, x_test, y_train, y_test = scale_data_with(select_scaler_by_parmeter(scaler), x_train, x_test, y_train, y_test)

    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))

    solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])

    batch_size = trial.suggest_int('batch_size', 1, 400)
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])

    clf = None

    if solver == 'sgd':
        learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2)
        shuffle = bool(trial.suggest_int('shuffle', 0, 1))
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        momentum = trial.suggest_uniform('momentum', 0.0, 1.0)
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)

        clf = MLPClassifier(
            random_state=42,
            max_iter=10000,
            hidden_layer_sizes=tuple(layers),
            solver=solver,
            batch_size=batch_size,
            activation=activation,

            shuffle=shuffle,
            learning_rate_init=learning_rate_init,
            learning_rate=learning_rate,
            momentum=momentum,
            alpha=alpha
        )

    elif solver == 'adam':
        shuffle = bool(trial.suggest_int('shuffle', 0, 1))
        learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2)
        beta_1 = trial.suggest_uniform('beta_1', 0.0, 1.0)
        beta_2 = trial.suggest_uniform('beta_2', 0.0, 1.0)
        epsilon = trial.suggest_loguniform('epsilon', 1e-5, 1e-2)

        clf = MLPClassifier(
            random_state=42,
            max_iter=10000,
            hidden_layer_sizes=tuple(layers),
            solver=solver,
            batch_size=batch_size,
            activation=activation,

            shuffle=shuffle,
            learning_rate_init=learning_rate_init,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon

        )

    else:
         clf = MLPClassifier(
            random_state=42,
            max_iter=10000,
            hidden_layer_sizes=tuple(layers),
            solver=solver,
            batch_size=batch_size,
            activation=activation
        )

    clusters = trial.suggest_int('clusters', 1, 50)

    pipeline = Pipeline([
        ('kmeans', KMeans(n_clusters=clusters)),
        ('clf', clf)
    ])

    pipeline.fit(x_train, y_train.values.ravel())

    predictions = pipeline.predict(x_test)

    matrix = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = matrix.ravel()
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)
    accuracy = (tn + tp) / (tn + tp + fp + fn)
    return accuracy


def optuna_search():

        if path.exists("study.pkl"):
            study = joblib.load('study.pkl')
        else:
            study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=20)

        joblib.dump(study, 'study.pkl')
        print('Optuna search done...')
        print('Best trial:')
        print(study.best_trial)
        print('Best params:')
        print(study.best_params)
        print('Best value:')
        print(study.best_value)


def objective_evo(trial):

    x_train, x_test, y_train, y_test = prepare_data_by_parameter('mean')

    #threshold = trial.suggest_uniform('threshold', 1.0, 4.0)
    #x_train, x_test, y_train, y_test = remove_outliers_zscore(threshold, x_train, x_test, y_train, y_test)

    x_train, x_test, y_train, y_test = scale_data_with(RobustScaler(), x_train, x_test, y_train, y_test)
    solver = 'adam'
    activation = 'tanh'

    #n_layers = trial.suggest_int('n_layers', 1, 5)
    layer_one = trial.suggest_int('layer_one', 1, 200)
    layer_two = trial.suggest_int('layer_two', 1, 200)
    layer_three = trial.suggest_int('layer_three', 1, 200)
    layer_four = trial.suggest_int('layer_four', 1, 200)
    #layer_five = trial.suggest_int('layer_five', 1, 200)

    layers = [layer_one, layer_two, layer_three, layer_four]

    #layers = [layer_one]

    # if n_layers >= 2:
    #     layers.append(layer_two)
    # if n_layers >= 3:
    #     layers.append(layer_three)
    # if n_layers >= 4:
    #     layers.append(layer_four)
    # if n_layers >= 5:
    #     layers.append(layer_five)

    batch_size = trial.suggest_int('batch_size', 5, 400)
    shuffle = bool(trial.suggest_int('shuffle', 0, 1))
    #learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2)
    learning_rate_init = trial.suggest_uniform('learning_rate_init', 1e-5, 1e-2)
    beta_1 = trial.suggest_uniform('beta_1', 0.0, 1.0)
    beta_2 = trial.suggest_uniform('beta_2', 0.0, 1.0)
    #epsilon = trial.suggest_loguniform('epsilon', 1e-5, 1e-2)
    epsilon = trial.suggest_uniform('epsilon', 1e-5, 1e-2)

    clf = MLPClassifier(
        random_state=42,
        max_iter=20000,
        hidden_layer_sizes=tuple(layers),
        solver=solver,
        batch_size=batch_size,
        activation=activation,

        shuffle=shuffle,
        learning_rate_init=learning_rate_init,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon

    )

    clusters = trial.suggest_int('clusters', 1, 50)
    pipeline = Pipeline([
        ('kmeans', KMeans(n_clusters=clusters)),
        ('clf', clf)
    ])

    pipeline.fit(x_train, y_train.values.ravel())

    predictions = pipeline.predict(x_test)

    matrix = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = matrix.ravel()
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)
    accuracy = (tn + tp) / (tn + tp + fp + fn)
    return accuracy


def optuna_search_evo():
    try:
        if path.exists("study_evo.pkl"):
            study = joblib.load('study_evo.pkl')
        else:
            study = optuna.create_study(direction='maximize', sampler=CmaEsSampler())

        study.optimize(objective_evo, n_trials=300)

    finally:
        joblib.dump(study, 'study_evo.pkl')
        print('Optuna search done...')
        print('Best trial:')
        print(study.best_trial)
        print('Best params:')
        print(study.best_params)
        print('Best value:')
        print(study.best_value)


def optuna_search_evo_param():
    try:
        if path.exists("study_evo_param.pkl"):
            study = joblib.load('study_evo_param.pkl')
        else:
            study = optuna.create_study(direction='maximize',)

        study.optimize(objective_evo, n_trials=500)

    finally:
        joblib.dump(study, 'study_evo_param.pkl')
        print('Optuna search done...')
        print('Best trial:')
        print(study.best_trial)
        print('Best params:')
        print(study.best_params)
        print('Best value:')
        print(study.best_value)


def best_parameters():
    xtrain, xtest, ytrain, ytest = prepare_data_mean()
    xtrain, xtest, ytrain, ytest = scale_data_with(StandardScaler(), xtrain, xtest, ytrain, ytest)

    # Best Accuracy: 81.0 %
    mlp = MLPClassifier(activation='logistic', batch_size=50, hidden_layer_sizes=(200, 200), max_iter=50000, random_state=42)
    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)


def best_from_optuna():
    xtrain, xtest, ytrain, ytest = prepare_data_mean()
    xtrain, xtest, ytrain, ytest = scale_data_with(RobustScaler(), xtrain, xtest, ytrain, ytest)

    # 0.82
    # {'missing_data_fix': 'mean', 'scaler': 'RobustScaler', 'n_layers': 4, 'n_units_0': 57, 'n_units_1': 79,
    # 'n_units_2': 68, 'n_units_3': 27, 'solver': 'adam', 'batch_size': 308, 'activation': 'tanh', 'shuffle': 0,
    # 'learning_rate_init': 9.46212720818499e-05, 'beta_1': 0.5259828543930881, 'beta_2': 0.9643476969943435, 'epsilon': 0.0012036632138694996}

    mlp = MLPClassifier(max_iter=10000,
                        random_state=42,
                        hidden_layer_sizes=(57, 79, 68, 27),
                        learning_rate_init=9.46212720818499e-05,
                        activation='tanh',
                        batch_size=308,
                        solver='adam',
                        shuffle=False,
                        beta_1=0.5259828543930881,
                        beta_2=0.9643476969943435,
                        epsilon=0.0012036632138694996
                        )
    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)


def best_from_optuna2():

    # {'missing_data_fix': 'drop', 'threshold': 2.3755166420810525, 'scaler': 'StandardScaler', 'n_layers': 1,
    #  'n_units_0': 56, 'solver': 'adam', 'batch_size': 157, 'activation': 'identity', 'shuffle': 0,
    #  'learning_rate_init': 0.004493807415028482, 'beta_1': 0.5048469093620177, 'beta_2': 0.140476025224519,
    #  'epsilon': 0.0025479671575892463}
    # f1:0.6933333333333334


    xtrain, xtest, ytrain, ytest = prepare_data_drop()
    xtrain, xtest, ytrain, ytest = remove_outliers_zscore(2.3755166420810525, xtrain, xtest, ytrain, ytest)
    xtrain, xtest, ytrain, ytest = scale_data_with(StandardScaler(), xtrain, xtest, ytrain, ytest)

    mlp = MLPClassifier(max_iter=10000,
                        random_state=42,
                        hidden_layer_sizes=(56),
                        learning_rate_init=0.004493807415028482,
                        activation='identity',
                        batch_size=157,
                        solver='adam',
                        shuffle=False,
                        beta_1=0.5048469093620177,
                        beta_2=0.140476025224519,
                        epsilon=0.0025479671575892463
                        )

    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)


def main():
    # TODO: Use pipeline?

    #best_from_optuna2()
    #best_from_optuna()
    #best_parameters()
    optuna_search()
    #optuna_search_evo()
    #optuna_search_evo_param()


if __name__ == '__main__':
    main()






