# Export model
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Models.
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

# Cross-Validation models.
from sklearn.model_selection import KFold

# Metrics
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

# Tempo de execução
import time
import os

random_state = 42


class FColors:
    WARNING = '\033[93m'
    OKGREEN = '\033[92m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def apply_KFold(classifier, x, y):
    # Configurando KFold
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)

    # Aplicando KFold nos modelos e calculando tempo decorrido
    mean_time = []
    for i in range(0, 3):
        ini = time.time()
        pred = cross_val_predict(classifier, x, y, n_jobs=-1, cv=kfold)
        final = time.time()
        mean_time.append(final - ini)

    return pred, np.round(np.mean(mean_time), 4)


def get_metrics(pred, mean_time, y):
    precision = precision_score(y, pred, average='macro')

    return (np.round(np.mean(precision) * 100, 1), mean_time)


def fill_report(report, classifier, precision, time):
    report['classifier'].append(classifier)
    report['precision'].append(precision)
    report['time'].append(time)


def run_test():
    arqIndex = 0
    with open('classifiers_tests.txt', 'w') as f:
        for arquivo in os.listdir(path='.\\DEV\\\DATA\\'):
            if arquivo.endswith('.csv'):
                print(arquivo)
                f.write('------------------------------\n\n')
                f.write(arquivo + '\n')
                df = pd.read_csv('.\\DEV\\\DATA\\' + arquivo)
                print(df.head())

                array = df.values
                data = array[:, 0:(len(df.columns)-1)]
                labels = array[:, (len(df.columns)-1)]

                report = {'classifier': [], 'precision': [], 'time': []}
#                 'bootstrap': False,
#  'max_depth': 20,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 1,
#  'min_samples_split': 4,
#  'n_estimators': 1000
                # Random Forest
                randomForest = RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=20,
                    max_features='sqrt',
                    min_samples_leaf=1,
                    min_samples_split=4,
                    bootstrap=False,
                    n_jobs=-1, random_state=random_state)

                pred, mean_time = apply_KFold(randomForest, data, labels)
                precision, time = get_metrics(
                    pred, mean_time, labels)
                fill_report(report, 'Random Forest', precision, time)

                # Decision Tree

                decisionTree = tree.DecisionTreeClassifier(
                    criterion='entropy', max_depth=7, max_features=None, splitter='best',
                    random_state=random_state)
                pred, mean_time = apply_KFold(decisionTree, data, labels)
                precision, time = get_metrics(pred, mean_time, labels)
                fill_report(report, 'Decision Tree', precision, time)

                # MLPClassifier
                # mlp = MLPClassifier(hidden_layer_sizes=(100, 45), random_state=random_state, solver='lbfgs', learning_rate='adaptive', alpha=0.00015, max_iter=500000)
                # pred, mean_time = apply_KFold(mlp, data, labels)
                # precision, time = get_metrics(
                #     pred, mean_time, labels)
                # fill_report(report, 'MLPClassifier', precision, time)
                #
                # # MultinomialNB
                # multinomialNB = MultinomialNB(alpha=49.7)
                # pred, mean_time = apply_KFold(multinomialNB, data, labels)
                # precision, time = get_metrics(
                #     pred, mean_time, labels)
                # fill_report(report, 'MultinomialNB', precision, time)

                report_df = pd.DataFrame(report)

                # Preenchendo o texto com os resultados
                for column in report_df.columns.values.tolist():
                    if(column != 'classifier'):
                        # f.write('\n\n{}\n'.format(column))
                        f.write('\n\\addplot coordinates {')
                        # if(column == 'time'):
                        f.write('\n')
                        for i in range(0, len(report['classifier'])):
                            f.write('({},{}) \n'.format(
                                report[column][i], arqIndex + 1))
                        # else:
                        #     for i in range(0, len(report['classifier'])):
                        #         f.write('({},{}) '.format(
                        #               report[column][i], arqIndex + 1))
                        f.write('};')
                f.write('\n\n------------------------------\n\n')
                arqIndex = arqIndex + 1
                # break
        f.close()


def export_model(model, data, labels):
    model.fit(data, labels)
    # Exporta o modelo em formato de arquivo
    joblib.dump(model, 'random_forest.sav')


if __name__ == '__main__':
    start_time = time.time()
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")

    run_test()

    end_time = time.time()
    print(f"{FColors.BOLD}{FColors.WARNING}{'time = '}{end_time - start_time}{' s'}{FColors.ENDC}\n")
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")
