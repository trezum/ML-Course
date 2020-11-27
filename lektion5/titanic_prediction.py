# (0 = died, 1 = survived)
# (1 = first class, 2 = second class, 3 = third class)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, make_scorer, \
    f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV


def prepare_data_drop():
    data_for_drop = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)

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

    # Scaling our data
    scaler = StandardScaler()
    scaler.fit(xtrain_drop)
    xtrain_drop = scaler.transform(xtrain_drop)
    scaler.fit(xtest_drop)
    xtest_drop = scaler.transform(xtest_drop)

    print(xtrain_drop)
    print(xtest_drop)
    print(yvalues_train_drop)
    print(yvalues_test_drop)
    return xtrain_drop, xtest_drop, yvalues_train_drop, yvalues_test_drop


def prepare_data_mean():
    data = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)
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

    # Scaling our data
    scaler = RobustScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def prepare_data_median():
    data = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)
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

    # Scaling our data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


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


def grid_search(xtrain, xtest, ytrain, ytest):

    #parameters = {'activation': ['logistic', 'tanh', 'relu'], 'batch_size': [50, 100, 200], 'hidden_layer_sizes': [(8, 8, 8), (100, 100)]}

    #parameters = {'activation': ['logistic'], 'batch_size': [1, 2, 5, 10, 20, 50], 'hidden_layer_sizes': [(8, 8, 8), (100, 100), (100, 100, 100), (200, 200, 200)]}
    parameters = {
        #'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'activation': ['logistic', 'relu'],
        'batch_size': [25, 50, 75, 100],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'hidden_layer_sizes': [[100, 100]],
        #'hidden_layer_sizes': [[a, a] for a in range(100, 400, 100)],
    }

    print(parameters)


    mlp_gs = MLPClassifier(random_state=42, max_iter=10000)
    scorer = make_scorer(f1_score)
    grid_obj = GridSearchCV(mlp_gs, parameters, scoring=scorer, verbose=2)
    grid_fit = grid_obj.fit(xtrain, ytrain.values.ravel())
    best_mlp = grid_fit.best_estimator_

    print_classifier_stats(best_mlp, xtrain, xtest, ytrain, ytest)
    print('Grid search done...')

def generate_networks():

    neuron_sizes = [25, 50, 100, 200, 300, 500]

    networks = []

    temparray = []

    for l in range(4):
        for n in neuron_sizes:
            for r in range(l):
                temparray.append(n)
                networks.append(tuple(temparray))
            temparray = []

    print(networks)
    return networks


def evo_search(xtrain, xtest, ytrain, ytest):
    layers = [[a, a] for a in range(10, 500, 100)]
    print(layers)

    parameters = {
                   'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  # 'solver': ['lbfgs', 'sgd', 'adam'],
                  # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'batch_size': [5, 10, 20, 50, 100],
                   'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                  # 'hidden_layer_sizes': generate_networks(),
                    'hidden_layer_sizes': layers
                  }
    print(parameters)
    print('Starting evolutionary search')

    cv = EvolutionaryAlgorithmSearchCV(estimator=MLPClassifier(random_state=42, max_iter=20000),
                                       params=parameters,
                                       scoring=make_scorer(f1_score),
                                       #cv=StratifiedKFold(n_splits=4),
                                       verbose=10,
                                       population_size=20,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=10,
                                       n_jobs=1)
    cv.fit(xtrain, ytrain.values.ravel())

    print_classifier_stats(cv.best_estimator_, xtrain, xtest, ytrain, ytest)
    print('Evo search done...')

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))

    x_train, x_test, y_train, y_test = prepare_data_mean()

    clf = MLPClassifier(hidden_layer_sizes=tuple(layers), max_iter=10000, random_state=42)
    clf.fit(x_train, y_train.values.ravel())

    predictions = clf.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    tn, fp, fn, tp = matrix.ravel()
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)

    accuracy = (tn + tp) / (tn + tp + fp + fn)

    #return clf.score(x_test, y_test)
    return accuracy


def optuna_search():
    #study = optuna.create_study(direction='maximize')
    study = optuna.create_study(direction='maximize', sampler='CmaEsSampler')
    study.optimize(objective, n_trials=50)
    print(study.best_params)
    print('Optuna search done...')


def main():

    #74% Data mean = MLPClassifier(activation='tanh', batch_size=50, hidden_layer_sizes=[300, 300], max_iter=10000, random_state=42)
    #73.0% Data median = MLPClassifier(activation='tanh', batch_size=50, hidden_layer_sizes=[300, 300], max_iter=10000, random_state=42)
    #65.789 Data drop = MLPClassifier(activation='logistic', batch_size=20, hidden_layer_sizes=[200, 200], max_iter=10000, random_state=42)
    xtrain, xtest, ytrain, ytest = prepare_data_mean()

    # print(data.describe(include='all'))
    # print(data.values)

    #mlp = MLPClassifier(activation='tanh', batch_size=10, hidden_layer_sizes=(300,), learning_rate_init=0.01, max_iter=10000, random_state=42)




    #Best Accuracy: 81.0 %
    mlp = MLPClassifier(activation='logistic', batch_size=50, hidden_layer_sizes=(200, 200), max_iter=50000, random_state=42)
    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)

    mlp = MLPClassifier(hidden_layer_sizes=(39), max_iter=50000, random_state=42)
    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)

    #75% med RobustScaler
    mlp = MLPClassifier(activation='logistic', batch_size=50, hidden_layer_sizes=[100, 100], max_iter=10000, random_state=42)
    mlp.fit(xtrain, ytrain.values.ravel())
    print_classifier_stats(mlp, xtrain, xtest, ytrain, ytest)

    #TODO: Data cleaning?
    #Seperat scalers?
    #scaler = StandardScaler()
    #scaler = RobustScaler()

    #Test forskællige scaler, clean og na kombinationer sammen?

    #grid_search(xtrain, xtest, ytrain, ytest)
    #evo_search(xtrain, xtest, ytrain, ytest)
    #optuna_search()


if __name__ == '__main__':
    main()






