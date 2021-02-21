import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

log_scale_list = ['max_depth', 'min_samples_leaf', 'min_samples_split', 'max_leaf_nodes',
                  'max_iter', 'alpha',
                  'n_estimators', 'learning_rate',
                  'C', 'max_iter',
                  ]


class model(object):
    def __init__(self, name, abbr, estimator, cv, params=None):
        self.name = name
        self.abbr = abbr
        self.estimator = estimator
        self.cv = cv
        self.params = params


def process_dataset(dataset_name):
    if dataset_name == "obesity":

        data = pd.read_csv(os.path.join("datasets", "ObesityDataSet_raw_and_data_sinthetic.csv"))

        data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
        data['family_history_with_overweight'] = data['family_history_with_overweight'].map({'no': 0, 'yes': 1})
        data['FAVC'] = data['FAVC'].map({'no': 0, 'yes': 1})
        data['CAEC'] = data['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        data['SMOKE'] = data['SMOKE'].map({'no': 0, 'yes': 1})
        data['SCC'] = data['SCC'].map({'no': 0, 'yes': 1})
        data['CALC'] = data['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        data['MTRANS'] = data['MTRANS'].map({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3,
                                             'Automobile': 4})
        data['NObeyesdad'] = data['NObeyesdad'].map({'Insufficient_Weight': 0,
                                                     'Normal_Weight': 1,
                                                     'Overweight_Level_I': 2,
                                                     'Overweight_Level_II': 3,
                                                     'Obesity_Type_I': 4,
                                                     'Obesity_Type_II': 5,
                                                     'Obesity_Type_III': 6})

    elif dataset_name == "online_shopping":

        data = pd.read_csv(os.path.join("datasets", "online_shoppers_intention.csv"))

        data['Month'] = data['Month'].map({'Feb': 2,
                                           'Mar': 3,
                                           'May': 5,
                                           'June': 6,
                                           'Jul': 7,
                                           'Aug': 8,
                                           'Sep': 9,
                                           'Oct': 10,
                                           'Nov': 11,
                                           'Dec': 12})
        data['VisitorType'] = data['VisitorType'].map({'Returning_Visitor': 0,
                                                       'New_Visitor': 1,
                                                       'Other': 2})
        data['Weekend'] = data['Weekend'].astype(int)
        data['Revenue'] = data['Revenue'].astype(int)

    else:
        data = []

    return data


def split_data(data, testing_raio=0.2, norm=False):
    data_matrix = data.values

    def scale(col, min, max):
        range = col.max() - col.min()
        a = (col - col.min()) / range
        return a * (max - min) + min

    if norm and data_matrix.shape[1] == 17:
        data_matrix[:, 1] = scale(data_matrix[:, 1], 0, 5)
        data_matrix[:, 2] = scale(data_matrix[:, 2], 0, 5)
        data_matrix[:, 3] = scale(data_matrix[:, 3], 0, 5)

    x = data_matrix[:, :-1]
    y = data_matrix[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_raio, shuffle=True,
                                                        random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # import and process datasets
    dataset_name_list = ["obesity", "online_shopping"]
    datas = []
    for dataset_name in dataset_name_list:
        datas.append(process_dataset(dataset_name))

    # Initialize list
    final_train_score_list = []
    final_test_score_list = []
    test_report = ''
    best_params_list = []

    # Define kfold
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)


    def get_hidden_layer_list(layer_num):
        param = []
        for i in range(1, layer_num + 1):
            out = []
            for j in range(i):
                out.append(32)
            param.append(tuple(out))
        return param


    for data_num, data in enumerate(datas):

        dataset_name = dataset_name_list[data_num]

        # Define classifiers configuration
        classifiers = [
            model(name='Decision Tree',
                  abbr='dt',
                  estimator=DecisionTreeClassifier(),
                  cv=kfold,
                  params={
                      'max_depth': [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
                      'min_samples_leaf': [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
                      'min_samples_split': [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
                      'max_leaf_nodes': [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
                      'max_features': list(range(1, len(data.columns))),
                  }),
            model(name='Neural Network',
                  abbr='mlp',
                  estimator=MLPClassifier(),
                  cv=kfold,
                  params={
                      'hidden_layer_sizes': get_hidden_layer_list(5),
                      'max_iter': [200, 400, 800, 1600, 3200],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs', 'sgd', 'adam'],
                      'alpha': np.logspace(-5, 1, 7),
                  }),
            model(name='Adaboost',
                  abbr='boost',
                  estimator=AdaBoostClassifier(),
                  cv=kfold,
                  params={
                      'n_estimators': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                      'learning_rate': np.logspace(-6, 1, 20)

                  }),
            model(name='SVM',
                  abbr='svm',
                  estimator=svm.SVC(),
                  cv=kfold,
                  params={
                      'C': [1 / 64., 1 / 32., 1 / 16., 1 / 8., 1 / 4., 1 / 2., 1, 2, 4, 8, 16, 32, 64, 128],
                      'max_iter': [200, 400, 800, 1600, 3200, 6400],
                      'kernel': ['linear', 'poly', 'rbf'],
                  }),
            model(name='KNN',
                  abbr='knn',
                  estimator=KNeighborsClassifier(),
                  cv=kfold,
                  params={
                      'n_neighbors': list(range(1, 33)),
                      'p': list(range(1, 10)),
                  }),
        ]

        # Split data
        if dataset_name == "obesity":
            x_train, x_test, y_train, y_test = split_data(data, norm=True)
        else:
            x_train, x_test, y_train, y_test = split_data(data)

        print("Training Set Shape: {}".format(x_train.shape))
        print("Testing Set Shape: {}".format(x_test.shape))

        for classifier in classifiers:

            ## Find Best Param

            # RandomizedSearchCV, quicker
            # grid_search = RandomizedSearchCV(estimator=classifier.estimator,
            #                                  param_distributions=classifier.params,
            #                                  scoring='accuracy',
            #                                  cv=kfold,
            #                                  n_jobs=-1,
            #                                  )

            # GridSearchCV, slower, but accurate
            grid_search = GridSearchCV(estimator=classifier.estimator,
                                       param_grid=classifier.params,
                                       scoring='accuracy',
                                       cv=kfold,
                                       n_jobs=-1,
                                       )

            # Get best param
            grid_search.fit(x_train, y_train)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_params_list.append(best_params)

            print('Best Params for "%s" Found by grid_search are:' % classifier.name)
            for param, value in best_params.items():
                print('%-32s: %s' % (param, value))
            print('\n')

            ## Learning Curve
            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=best_estimator,
                                                                                  X=x_train,
                                                                                  y=y_train,
                                                                                  train_sizes=np.linspace(0.1, 1.0, 10),
                                                                                  n_jobs=-1,
                                                                                  return_times=True,
                                                                                  cv=kfold,
                                                                                  )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            # Plot learning curve
            # Cross Validation
            plt.title("Learning Curve for %s - %s" % (classifier.name, dataset_name))
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.legend(loc="best")
            plt.savefig('images/learning_curve_%s_%s' % (classifier.abbr, dataset_name))
            plt.show()

            # Plot Scalability
            plt.title("Scalability for %s - %s" % (classifier.name, dataset_name))
            plt.xlabel("Training examples")
            plt.ylabel("fit_times")
            plt.grid()

            plt.plot(train_sizes, fit_times_mean, 'o-')
            plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
            plt.savefig('images/scalability_%s_%s' % (classifier.abbr, dataset_name))
            plt.show()

            # Plot fit_time vs score
            plt.title("Performance for %s - %s" % (classifier.name, dataset_name))
            plt.xlabel("fit_times")
            plt.ylabel("Score")
            plt.grid()
            plt.plot(fit_times_mean, test_scores_mean, 'o-')
            plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
            plt.savefig('images/performance_%s_%s' % (classifier.abbr, dataset_name))
            plt.show()

            ## Validation Curve
            for param_name, param_range in classifier.params.items():
                train_scores, test_scores = validation_curve(estimator=best_estimator,
                                                             X=x_train,
                                                             y=y_train,
                                                             cv=kfold,
                                                             n_jobs=-1,
                                                             param_name=param_name,
                                                             param_range=param_range)
                if isinstance(param_range[0], tuple):
                    param_range = [len(p) for p in param_range]
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                plt.title("Validation Curve for %s - %s" % (classifier.name, dataset_name))
                plt.xlabel(param_name)
                plt.ylabel("Score")
                lw = 2
                plt.plot(param_range, train_scores_mean, 'o-', label="Training score",
                         color="darkorange", lw=lw)
                plt.fill_between(param_range, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
                         color="navy", lw=lw)

                if param_name in log_scale_list:
                    plt.xscale('log', base=2)
                plt.fill_between(param_range, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.2,
                                 color="navy", lw=lw)
                plt.legend(loc="best")
                plt.savefig('images/validation_%s_%s_%s' % (classifier.abbr, param_name, dataset_name))
                plt.show()

            ## Final Test Score Calculation
            best_estimator.fit(x_train, y_train)
            train_predict = best_estimator.predict(x_train)
            test_predict = best_estimator.predict(x_test)
            final_train_score = accuracy_score(y_train, train_predict)
            final_test_score = accuracy_score(y_test, test_predict)
            final_train_score_list.append(final_train_score)
            final_test_score_list.append(final_test_score)
            print('%s - %s final train score: %s' % (dataset_name, classifier.name, final_train_score))
            print('%s - %s final test score:  %s' % (dataset_name, classifier.name, final_test_score))
            test_report += '%s - %s final train score: %s\n' % (dataset_name, classifier.name, final_train_score)
            test_report += '%s - %s final test score:  %s\n' % (dataset_name, classifier.name, final_test_score)

        # Final Test Score Plot
        classifier_list = [classifier.name for classifier in classifiers]
        plt.title("Final Test Score Plot - %s" % dataset_name)
        plt.xlabel('Classifier Name')
        plt.ylabel("Score")
        lw = 2
        plt.plot(classifier_list, final_train_score_list[5 * data_num:5 * (data_num + 1)], 'o-',
                 label="Final Training score",
                 color="darkorange", lw=lw)
        plt.plot(classifier_list, final_test_score_list[5 * data_num:5 * (data_num + 1)], 'o-',
                 label="Final Testing score",
                 color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig('images/final_test_score_%s' % dataset_name)
        plt.show()

    # Save final test report
    text_file = open("test_report.txt", "w")
    text_file.write(test_report)
    text_file.close()
