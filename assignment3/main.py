import os
import six
import sys
sys.modules['sklearn.externals.six'] = six

import pandas as pd
from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks, Queens, MaxKColor, Knapsack, SixPeaks


import numpy as np
import matplotlib.pyplot as plt
import time
import utils

from mlrose.decay import ExpDecay
from mlrose.neural import NeuralNetwork

from sklearn.metrics import log_loss, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.decay import ExpDecay


from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score


IMAGE_DIR = 'images/'

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

    # for obesity
    if norm and data_matrix.shape[1] == 17:
        data_matrix[:, 1] = scale(data_matrix[:, 1], 0, 5)
        data_matrix[:, 2] = scale(data_matrix[:, 2], 0, 5)
        data_matrix[:, 3] = scale(data_matrix[:, 3], 0, 5)

    x = data_matrix[:, :-1]
    y = data_matrix[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_raio, shuffle=True,
                                                        random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


def kmean(data_x, data_y, range_n_clusters, rounds, plot=True, raw_data=True, label=''):
    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()
        data_y = data_y.to_numpy()

    nmis = []
    sil_avgs = []

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 4, 1)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        plt.xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        plt.ylim([0, len(data_x) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(data_x)
        cluster_labels = clusterer.labels_
        print("NMI score: %.6f" % normalized_mutual_info_score(data_y, cluster_labels))
        nmis.append(normalized_mutual_info_score(data_y, cluster_labels))

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data_x, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        sil_avgs.append(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data_x, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            cmap = cm.get_cmap("Spectral")
            color = cmap((float(i) + 0.5) / n_clusters)

            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        plt.title("The silhouette plot for the various clusters.")
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        cmap = cm.get_cmap("Spectral")
        colors = cmap((cluster_labels.astype(float) + 0.5) / n_clusters)


        for i, selected in enumerate(rounds):
            plt.subplot(1, 4, i + 2)
            plt.scatter(data_x[:, selected[0]], data_x[:, selected[1]], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_

            # Draw white circles at cluster centers
            plt.scatter(centers[:, selected[0]], centers[:, selected[1]], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                plt.scatter(c[selected[0]], c[selected[1]], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            if raw_data:
                plt.title("{} vs {}".format(features[selected[0]], features[selected[1]]))
                plt.xlabel("Feature space, feature {} \"{}\"".format(selected[0], features[selected[0]]))
                plt.ylabel("Feature space, feature {} \"{}\"".format(selected[1], features[selected[1]]))
            else:
                plt.xlabel("Feature space, Transformed feature {}".format(selected[0]))
                plt.ylabel("Feature space, Transformed feature {}".format(selected[1]))

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        if label == '':
            plt.savefig(IMAGE_DIR + 'kmeans_{}_{}'.format(n_clusters, dataset_name))
        else:
            plt.savefig(IMAGE_DIR + 'kmeans_{}_{}_{}'.format(label, n_clusters, dataset_name))

        if plot:
            plt.show()

    if len(range_n_clusters) > 1:
        plt.figure()
        plt.plot(range_n_clusters, nmis, 'o-', color="r", label='Normalized Mutual Info')
        plt.plot(range_n_clusters, sil_avgs, 'o-', color="g", label='Silhouette Average')
        plt.title('KMeans Evaluation')
        plt.xlabel("Cluster Number")
        plt.ylabel("Score")
        plt.legend(loc='best')
        if label == '':
            plt.savefig(IMAGE_DIR + 'kmeans_evaluation_{}'.format(dataset_name))
        else:
            plt.savefig(IMAGE_DIR + 'kmeans_evaluation_{}_{}'.format(label, dataset_name))

        plt.show()

    return clusterer


def em(data_x, data_y, range_n_clusters, rounds, plot=True, raw_data=True, label=''):

    bics = []
    aics = []
    nmis = []

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()
        data_y = data_y.to_numpy()

    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(data_x)
        cluster_labels = clusterer.predict(data_x)
        print("NMI score: %.6f" % normalized_mutual_info_score(data_y, cluster_labels))
        nmis.append(normalized_mutual_info_score(data_y, cluster_labels))
        bics.append(clusterer.bic(data_x))
        aics.append(clusterer.aic(data_x))

        # 2nd Plot showing the actual clusters formed
        cmap = cm.get_cmap("Spectral")
        colors = cmap((cluster_labels.astype(float) + 0.5) / n_clusters)

        plt.figure(figsize=(12, 4))

        for i, selected in enumerate(rounds):
            plt.subplot(1, 3, i + 1)

            plt.scatter(data_x[:, selected[0]], data_x[:, selected[1]], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.means_

            # Draw white circles at cluster centers
            plt.scatter(centers[:, selected[0]], centers[:, selected[1]], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                plt.scatter(c[selected[0]], c[selected[1]], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
            if raw_data:
                plt.title("{} vs {}".format(features[selected[0]], features[selected[1]]))
                plt.xlabel("Feature space, feature {} \"{}\"".format(selected[0], features[selected[0]]))
                plt.ylabel("Feature space, feature {} \"{}\"".format(selected[1], features[selected[1]]))
            else:
                plt.xlabel("Feature space, Transformed feature {}".format(selected[0]))
                plt.ylabel("Feature space, Transformed feature {}".format(selected[1]))


        plt.suptitle(("Clusters plot for EM clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        if label == '':
            plt.savefig(IMAGE_DIR + 'em_{}_{}'.format(n_clusters, dataset_name))
        else:
            plt.savefig(IMAGE_DIR + 'em_{}_{}_{}'.format(label, n_clusters, dataset_name))

        if plot:
            plt.show()

    if len(range_n_clusters) > 1:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(range_n_clusters, aics, 'o-', color="r", label='AIC Score')
        ax1.plot(range_n_clusters, bics, 'o-', color="g", label='BIC Score')
        ax1.set_title('EM Evaluation')
        ax1.set_xlabel("Cluster Number")
        ax1.set_ylabel("AIC/BIC Score")
        ax1.legend(loc='upper left')
        # plt.savefig(IMAGE_DIR + 'em_evaluation_{}'.format(dataset_name))
        # plt.show()

        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(range_n_clusters, nmis, 'o-', color="b", label='NMI Score')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel("NMI Score")
        ax2.legend(loc='upper right')

        if label == '':
            plt.savefig(IMAGE_DIR + 'em_evaluation_{}'.format(dataset_name))
        else:
            plt.savefig(IMAGE_DIR + 'em_evaluation_{}_{}'.format(label, dataset_name))

        plt.show()



############################################################################
#########################BEGIN EXPERIMENT###################################
############################################################################


# import and process datasets
# dataset_name_list = ["obesity", "online_shopping"]
dataset_name_list = ["obesity"]
# dataset_name_list = ["online_shopping"]
datas = []
for dataset_name in dataset_name_list:
    datas.append(process_dataset(dataset_name))

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for data_num, data in enumerate(datas):
    dataset_name = dataset_name_list[data_num]
    # Split data
    if dataset_name == "obesity":
        x_train, x_test, y_train, y_test = split_data(data, norm=True)
        data_x = data.iloc[:, :16]
        data_y = data.iloc[:, 16]
    else:
        x_train, x_test, y_train, y_test = split_data(data, norm=False)
        data_x = data.iloc[:, :17]
        data_y = data.iloc[:, 17]

    features = list(data_x.columns.values)

    # scaler = MinMaxScaler(feature_range=[0, 100])
    # scaler.fit(data_x)
    # X_norm = pd.DataFrame(scaler.transform(data_x))
    # data_x = X_norm


    print("Training Set Shape: {}".format(x_train.shape))
    print("Testing Set Shape: {}".format(x_test.shape))


    ##########################################
    # Feature importance
    ##########################################
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    top_10 = []
    top_10_vals = []
    top_10_idx = []
    # for f, g in zip(range(x_train.shape[1]), indices):
    for f, g in zip(range(x_train.shape[1]), indices[:10]):
        print("%2d) % -*s %f" % (f + 1, 30, data.columns[indices[f]], importances[indices[f]]))
        top_10.append(data.columns[indices[f]])
        top_10_idx.append(indices[f])
        top_10_vals.append(importances[indices[f]])

    plt.title('Feature Importance')
    plt.bar(top_10, top_10_vals, align='center')
    plt.xticks(top_10, rotation=90)
    plt.tight_layout()
    # plt.show()
    # filename = '%s_%s' % ('features', dataset.__class__.__name__)
    # chart_path = 'report/images/%s.png' % filename
    # plt.savefig(chart_path)
    plt.savefig(IMAGE_DIR + 'feature_importance_{}'.format(dataset_name))
    plt.show()


    ##########################################
    # k-mean
    ##########################################

    # range_n_clusters = [4]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(3, 2), (3, 1), (2, 1)]
    kmean(data_x, data_y, range_n_clusters, rounds)

    ##########################################
    # EM
    ##########################################
    # range_n_clusters = [4]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(3, 2), (3, 1), (2, 1)]
    em(data_x, data_y, range_n_clusters, rounds)


    ##########################################
    # PCA feature transformation
    ##########################################

    pca = PCA(n_components=16, random_state=10)
    # X_r = pca.fit(data_x).transform(data_x)
    X_r = pca.fit_transform(data_x)
    X_pca = X_r
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure(figsize=(18, 4))
    # plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2
    plt.subplot(1, 4, 1)

    for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6]):
        plt.scatter(X_r[data_y == i, 0], X_r[data_y == i, 1], color=color, alpha=.8, lw=lw, label='Label {}'.format(i))
        plt.xlabel("Transformed PCA Component 1")
        plt.ylabel("Transformed PCA Component 2")

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Obesity dataset')

    pca_components = [0, X_pca.shape[1] - 1]

    for i, comp in enumerate(pca_components):
        plt.subplot(1, 4, i + 2)
        plt.hist(X_r[data_y == 0, comp], density=True, alpha=0.5, color=colors[0], bins=30, label='Label 0')
        plt.axvline(np.median(X_r[data_y == 0, comp]), color='k', linestyle='dashed', linewidth=1)
        plt.hist(X_r[data_y == 4, comp], density=True, alpha=0.5, color=colors[1], bins=30, label='Label 3')
        plt.axvline(np.median(X_r[data_y == 3, comp]), color='k', linestyle='dashed', linewidth=1)
        plt.hist(X_r[data_y == 6, comp], density=True, alpha=0.5, color=colors[2], bins=30, label='Label 6')
        plt.axvline(np.median(X_r[data_y == 6, comp]), color='k', linestyle='dashed', linewidth=1)
        plt.title('PCA Component {}'.format(comp + 1))
        plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(list(range(1, len(pca.explained_variance_ratio_) + 1)), np.cumsum(pca.explained_variance_ratio_), '-o')
    plt.title('Cumulative % of Variance Explained by Components')
    plt.xlabel("No. of Components")
    plt.ylabel("Cumulative % of Variance")
    plt.savefig(IMAGE_DIR + 'pca_evaluation_{}'.format(dataset_name))
    plt.show()


    ##########################################
    # ICA feature transformation
    ##########################################

    from scipy.stats import kurtosis, skew

    plt.figure(figsize=(18, 4))

    avg_kurt = []
    rc_error = []
    for component in range(1, 17):
        # ICA feature transformation
        ica = FastICA(n_components=component, random_state=10)
        X_r = ica.fit_transform(data_x)
        X_ica = X_r

        avg_kurt.append(np.mean(np.apply_along_axis(kurtosis, 0, X_ica)))

        X_reconstructed = ica.inverse_transform(X_r)
        rc_error.append(((data_x - X_reconstructed) ** 2).mean().sum())

    # plt.plot(list(range(1, len(rc_error) + 1)), rc_error, '-o')
    # plt.show()


    ica = FastICA(n_components=15, random_state=10)
    X_r = ica.fit_transform(data_x)
    X_ica = X_r
    kurtosis = np.apply_along_axis(kurtosis, 0, X_ica)

    colors = ["b","g","r","c","m","y","k"]
    lw = 2
    plt.subplot(1, 4, 1)

    d = 0
    e = 4
    for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6]):
        plt.scatter(X_r[data_y == i, d], X_r[data_y == i, e], color=color, alpha=.8, lw=lw, label='Label {}'.format(i))
    plt.xlabel("Transformed ICA Component 1")
    plt.ylabel("Transformed ICA Component 2")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('ICA of Obesity dataset')
    # plt.show()

    plt.subplot(1, 4, 2)
    plt.title('Choosing number of components')
    plt.xlabel('No. of components')
    plt.ylabel('Average Kurtosis across IC')
    plt.plot(list(range(1, len(avg_kurt) + 1)), avg_kurt, '-o')
    plt.xticks(list(range(1, len(avg_kurt) + 1)))


    # ICA feature transformation
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    plt.subplot(1, 4, 3)

    feature = 0
    plt.hist(X_r[data_y == 0, feature], density=True, alpha=0.5, color=colors[0], bins=30, label='Label 0')
    plt.axvline(np.median(X_r[data_y == 0, feature]), color='k', linestyle='dashed', linewidth=1)
    plt.hist(X_r[data_y == 3, feature], density=True, alpha=0.5, color=colors[1], bins=30, label='Label 3')
    plt.axvline(np.median(X_r[data_y == 3, feature]), color='k', linestyle='dashed', linewidth=1)
    plt.hist(X_r[data_y == 6, feature], density=True, alpha=0.5, color=colors[2], bins=30, label='Label 6')
    plt.axvline(np.median(X_r[data_y == 6, feature]), color='k', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.title('ICA Component 1')

    import numpy as np


    plt.subplot(1, 4, 4)
    plt.title('Kurtosis Distribution (n=16)')
    plt.scatter(list(range(1, len(kurtosis) + 1)), kurtosis, c='g')
    plt.ylabel('Kurtosis Value')
    plt.xlabel('Component No.')
    plt.xticks(list(range(1, len(kurtosis) + 1)))

    plt.savefig(IMAGE_DIR + 'ica_evaluation_{}'.format(dataset_name))
    plt.show()

    ##########################################
    # Random Projection feature transformation
    ##########################################

    states = [10, 13, 16, 19]
    plt.figure(figsize=(18, 4))

    for i, state in enumerate(states):

        rca = GaussianRandomProjection(n_components=16, random_state=state)
        X_r = rca.fit_transform(data_x)
        X_rca = X_r

        plt.subplot(1, 4, i + 1)
        colors = ["b","g","r","c","m","y","k"]
        lw = 2

        for color, cp in zip(colors, [0, 1, 2, 3, 4, 5, 6]):
            plt.scatter(X_r[data_y == cp, 0], X_r[data_y == cp, 1], color=color, alpha=.8, lw=lw, label='Label {}'.format(cp))
        plt.xlabel("Transformed RCA Component 1")
        plt.ylabel("Transformed RCA Component 2")
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('Random Projection of Obesity dataset, Try No. {}'.format(i + 1))
    plt.savefig(IMAGE_DIR + 'rca_evaluation_{}'.format(dataset_name))
    plt.show()


    ##########################################
    # Univariate feature selection (K best)
    ##########################################

    from sklearn.feature_selection import chi2

    X_new = SelectKBest(chi2, k=4).fit_transform(data_x, data_y)
    X_fs = X_new

    plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6]):
        plt.scatter(X_new[data_y == i, 1], X_new[data_y == i, 2], color=color, alpha=.8, lw=lw, label='Label {}'.format(i))
    plt.xlabel("Transformed KBest Component 2")
    plt.ylabel("Transformed KBest Component 3")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Chi square feature selection of Obesity dataset')
    plt.savefig(IMAGE_DIR + 'kbest_evaluation_{}'.format(dataset_name))
    plt.show()




    ##########################################
    # k-mean
    ##########################################

    # PCA
    pca = PCA(n_components=4, random_state=10)
    X_r = pca.fit_transform(data_x)
    X_pca = X_r
    # range_n_clusters = [3]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_pca = kmean(X_pca, data_y, range_n_clusters, rounds, raw_data=False, label='pca')

    # ICA
    ica = FastICA(n_components=6, random_state=10)
    X_r = ica.fit_transform(data_x)
    X_ica = X_r
    # range_n_clusters = [6]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_ica = kmean(X_ica, data_y, range_n_clusters, rounds, raw_data=False, label='ica')

    # RCA
    rca = GaussianRandomProjection(n_components=4, random_state=10)
    X_r = rca.fit_transform(data_x)
    X_rca = X_r
    # range_n_clusters = [3]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_rca = kmean(X_rca, data_y, range_n_clusters, rounds, raw_data=False, label='rca')

    # KBest
    X_new = SelectKBest(chi2, k=4).fit_transform(data_x, data_y)
    X_fs = X_new
    # range_n_clusters = [3]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(2, 0), (2, 1), (2, 3)]
    cluster_kbest = kmean(X_fs, data_y, range_n_clusters, rounds, raw_data=False, label='kbest')

    ##########################################
    # EM
    ##########################################

    # PCA
    # range_n_clusters = [8]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_pca, data_y, range_n_clusters, rounds, raw_data=False, label='pca')

    # ICA
    # range_n_clusters = [8]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_ica, data_y, range_n_clusters, rounds, raw_data=False, label='ica')

    # RCA
    # range_n_clusters = [8]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_rca, data_y, range_n_clusters, rounds, raw_data=False, label='rca')

    # KBest
    # range_n_clusters = [8]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rounds = [(2, 0), (2, 1), (2, 3)]
    em(X_fs, data_y, range_n_clusters, rounds, raw_data=False, label='kbest')






    ##########################################
    # Rerun ANN on transformed features
    ##########################################

    # PCA
    pca = PCA(n_components=4, random_state=10)
    X_r = pca.fit_transform(data_x)
    X_pca = X_r
    range_n_clusters = [6]
    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_pca = kmean(X_pca, data_y, range_n_clusters, rounds, plot=False)

    ica = FastICA(n_components=15, random_state=10)
    X_r = ica.fit_transform(data_x)
    X_ica = X_r
    range_n_clusters = [6]
    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_ica = kmean(X_ica, data_y, range_n_clusters, rounds, plot=False)

    rca = GaussianRandomProjection(n_components=4, random_state=10)
    X_r = rca.fit_transform(data_x)
    X_rca = X_r
    range_n_clusters = [7]
    rounds = [(0, 1), (0, 2), (1, 2)]
    cluster_rca = kmean(X_rca, data_y, range_n_clusters, rounds, plot=False)

    X_new = SelectKBest(chi2, k=4).fit_transform(data_x, data_y)
    X_fs = X_new
    range_n_clusters = [6]
    rounds = [(2, 0), (2, 1), (2, 3)]
    cluster_kbest = kmean(X_fs, data_y, range_n_clusters, rounds, plot=False)



    # Reconstruct data
    xpca_train, xpca_test, ypca_train, ypca_test = train_test_split(X_pca, data_y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    stratify=data_y)
    xica_train, xica_test, yica_train, yica_test = train_test_split(X_ica, data_y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    stratify=data_y)
    xrca_train, xrca_test, yrca_train, yrca_test = train_test_split(X_rca, data_y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    stratify=data_y)
    xkbest_train, xkbest_test, ykbest_train, ykbest_test = train_test_split(X_fs, data_y,
                                                                            test_size=0.2,
                                                                            shuffle=True,
                                                                            random_state=42,
                                                                            stratify=data_y)

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import learning_curve

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        # plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                        random_state=0,
                        alpha=0.001,
                        max_iter=3200,
                        activation="tanh",
                        solver="lbfgs")

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt = plot_learning_curve(clf, "MLP using PCA transformed features", xpca_train, ypca_train, ylim=[0,1])
    plt.subplot(1, 4, 2)
    plt = plot_learning_curve(clf, "MLP using ICA transformed features", xica_train, yica_train, ylim=[0,1])
    plt.subplot(1, 4, 3)
    plt = plot_learning_curve(clf, "MLP using RCA transformed features", xrca_train, yrca_train, ylim=[0,1])
    plt.subplot(1, 4, 4)
    plt = plot_learning_curve(clf, "MLP using KBest transformed features", xkbest_train, ykbest_train, ylim=[0,1])
    plt.savefig(IMAGE_DIR + 'nn_on_dr_data')
    plt.show()

    clf.fit(xpca_train, ypca_train)
    report = classification_report(ypca_test, clf.predict(xpca_test))
    print('NN using PCA transformed features  test classification report = \n {}'.format(report))

    clf.fit(xica_train, yica_train)
    report = classification_report(yica_test, clf.predict(xica_test))
    print('NN using ICA transformed features test classification report = \n {}'.format(report))

    clf.fit(xrca_train, yrca_train)
    report = classification_report(yrca_test, clf.predict(xrca_test))
    print('NN using RCA transformed features test classification report = \n {}'.format(report))

    clf.fit(xkbest_train, ykbest_train)
    report = classification_report(ykbest_test, clf.predict(xkbest_test))
    print('NN using KBest transformed features test classification report = \n {}'.format(report))


    #################################################
    #Rerun ANN on transformed features with clusters new feature

    # clf = MLPClassifier(hidden_layer_sizes=(20, 5), random_state=0, solver="lbfgs")

    # Reconstruction for task 5
    cluster_label_pca = cluster_pca.labels_
    X_df = pd.DataFrame(X_pca)
    X_df[X_df.shape[1]] = cluster_label_pca
    xpca_w_label_train, xpca_w_label_test, ypca_w_label_train, ypca_w_label_test = train_test_split(X_df, data_y,
                                                                                                    test_size=0.2,
                                                                                                    shuffle=True,
                                                                                                    random_state=42,
                                                                                                    stratify=data_y)

    cluster_label_ica = cluster_ica.labels_
    X_df = pd.DataFrame(X_ica)
    X_df[X_df.shape[1]] = cluster_label_ica

    xica_w_label_train, xica_w_label_test, yica_w_label_train, yica_w_label_test = train_test_split(X_df, data_y,
                                                                                                    test_size=0.2,
                                                                                                    shuffle=True,
                                                                                                    random_state=42,
                                                                                                    stratify=data_y)

    cluster_label_rca = cluster_rca.labels_
    X_df = pd.DataFrame(X_rca)
    X_df[X_df.shape[1]] = cluster_label_rca

    xrca_w_label_train, xrca_w_label_test, yrca_w_label_train, yrca_w_label_test = train_test_split(X_df, data_y,
                                                                                                    test_size=0.2,
                                                                                                    shuffle=True,
                                                                                                    random_state=42,
                                                                                                    stratify=data_y)
    cluster_label_kbest = cluster_kbest.labels_
    X_df = pd.DataFrame(X_fs)
    X_df[X_df.shape[1]] = cluster_label_kbest

    xkbest_w_label_train, xkbest_w_label_test, ykbest_w_label_train, ykbest_w_label_test = train_test_split(X_df, data_y,
                                                                                                            test_size=0.2,
                                                                                                            shuffle=True,
                                                                                                            random_state=42,
                                                                                                            stratify=data_y)

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt = plot_learning_curve(clf, "MLP using PCA transformed features with label", xpca_w_label_train, ypca_w_label_train, ylim=[0,1])
    plt.subplot(1, 4, 2)
    plt = plot_learning_curve(clf, "MLP using ICA transformed features with label", xica_w_label_train, yica_w_label_train, ylim=[0,1])
    plt.subplot(1, 4, 3)
    plt = plot_learning_curve(clf, "MLP using RCA transformed features with label", xrca_w_label_train, yrca_w_label_train, ylim=[0,1])
    plt.subplot(1, 4, 4)
    plt = plot_learning_curve(clf, "MLP using KBest transformed features with label", xkbest_w_label_train, ykbest_w_label_train, ylim=[0,1])
    plt.savefig(IMAGE_DIR + 'nn_on_dr_data_w_label')
    plt.show()

    clf.fit(xpca_w_label_train, ypca_w_label_train)
    report = classification_report(ypca_w_label_test, clf.predict(xpca_w_label_test))
    print('NN using PCA transformed features with label test classification report = \n {}'.format(report))

    clf.fit(xica_w_label_train, yica_w_label_train)
    report = classification_report(yica_w_label_test, clf.predict(xica_w_label_test))
    print('NN using ICA transformed features with label test classification report = \n {}'.format(report))

    clf.fit(xrca_w_label_train, yrca_w_label_train)
    report = classification_report(yrca_w_label_test, clf.predict(xrca_w_label_test))
    print('NN using RCA transformed features with label test classification report = \n {}'.format(report))

    clf.fit(xkbest_w_label_train, ykbest_w_label_train)
    report = classification_report(ykbest_w_label_test, clf.predict(xkbest_w_label_test))
    print('NN using KBest transformed features with label test classification report = \n {}'.format(report))







    clf.fit(x_train, y_train)
    plt = plot_learning_curve(clf, "MLP using ICA transformed features with label", x_train, y_train, ylim=[0,1])
    plt.show()

    report = classification_report(y_test, clf.predict(x_test))
    print('NN using KBest transformed features with label test classification report = \n {}'.format(report))

    # plot_learning_curve(clf, "MLP using PCA transformed features", X_df, y, ylim=[0,1])
    #
    # clusterer = KMeans(n_clusters=6, random_state=10).fit(X_ica)
    # y_kmeans = clusterer.labels_
    # X_df = pd.DataFrame(X_ica)
    # X_df[11] = y_kmeans
    # plot_learning_curve(clf, "MLP using ICA transformed features", X_df, y, ylim=[0,1])
    #
    # clusterer = KMeans(n_clusters=6, random_state=10).fit(X_rca)
    # y_kmeans = clusterer.labels_
    # X_df = pd.DataFrame(X_rca)
    # X_df[11] = y_kmeans
    # plot_learning_curve(clf, "MLP using RCA transformed features", X_df, y, ylim=[0,1])
    #
    # clusterer = KMeans(n_clusters=6, random_state=10).fit(X_fs)
    # y_kmeans = clusterer.labels_
    # X_df = pd.DataFrame(X_fs)
    # X_df[11] = y_kmeans
    # plot_learning_curve(clf, "MLP using selected 5 features", X_df, y, ylim=[0,1])























############################################################################
#########################BEGIN EXPERIMENT 2#################################
############################################################################






# import and process datasets
# dataset_name_list = ["obesity", "online_shopping"]
dataset_name_list = ["online_shopping"]
# dataset_name_list = ["online_shopping"]
datas = []
for dataset_name in dataset_name_list:
    datas.append(process_dataset(dataset_name))

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for data_num, data in enumerate(datas):
    dataset_name = dataset_name_list[data_num]
    # Split data
    if dataset_name == "obesity":
        x_train, x_test, y_train, y_test = split_data(data, norm=True)
        data_x = data.iloc[:, :16]
        data_y = data.iloc[:, 16]
    else:
        x_train, x_test, y_train, y_test = split_data(data, norm=False)
        data_x = data.iloc[:, :17]
        data_y = data.iloc[:, 17]

    features = list(data_x.columns.values)

    # scaler = MinMaxScaler(feature_range=[0, 100])
    # scaler.fit(data_x)
    # X_norm = pd.DataFrame(scaler.transform(data_x))
    # data_x = X_norm

    print("Training Set Shape: {}".format(x_train.shape))
    print("Testing Set Shape: {}".format(x_test.shape))

    ##########################################
    # Feature importance
    ##########################################
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    top_10 = []
    top_10_vals = []
    top_10_idx = []
    # for f, g in zip(range(x_train.shape[1]), indices):
    for f, g in zip(range(x_train.shape[1]), indices[:10]):
        print("%2d) % -*s %f" % (f + 1, 30, data.columns[indices[f]], importances[indices[f]]))
        top_10.append(data.columns[indices[f]])
        top_10_idx.append(indices[f])
        top_10_vals.append(importances[indices[f]])

    plt.title('Feature Importance')
    plt.bar(top_10, top_10_vals, align='center')
    plt.xticks(top_10, rotation=90)
    plt.tight_layout()
    # plt.show()
    # filename = '%s_%s' % ('features', dataset.__class__.__name__)
    # chart_path = 'report/images/%s.png' % filename
    plt.savefig(IMAGE_DIR + 'feature_importance_{}'.format(dataset_name))
    plt.show()

    ##########################################
    # k-mean
    ##########################################

    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(8, 7), (8, 5), (7, 5)]
    kmean(data_x, data_y, range_n_clusters, rounds)

    ##########################################
    # EM
    ##########################################
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(8, 7), (8, 5), (7, 5)]
    em(data_x, data_y, range_n_clusters, rounds)

    ##########################################
    # PCA feature transformation
    ##########################################

    pca = PCA(n_components=17, random_state=10)
    # X_r = pca.fit(data_x).transform(data_x)
    X_r = pca.fit_transform(data_x)
    X_pca = X_r
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure(figsize=(18, 4))
    # plt.figure()
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    lw = 2
    plt.subplot(1, 4, 1)

    for color, i in zip(colors, [0, 1]):
        plt.scatter(X_r[data_y == i, 0], X_r[data_y == i, 1], color=color, alpha=.8, lw=lw,
                    label='Label {}'.format(i))
        plt.xlabel("Transformed PCA Component 1")
        plt.ylabel("Transformed PCA Component 2")

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Obesity dataset')

    pca_components = [0, X_pca.shape[1] - 1]

    for i, comp in enumerate(pca_components):
        plt.subplot(1, 4, i + 2)
        plt.hist(X_r[data_y == 0, comp], density=True, alpha=0.5, color=colors[0], bins=50, label='Label 0')
        plt.axvline(np.median(X_r[data_y == 0, comp]), color='k', linestyle='dashed', linewidth=1)
        plt.hist(X_r[data_y == 1, comp], density=True, alpha=0.5, color=colors[1], bins=50, label='Label 1')
        plt.axvline(np.median(X_r[data_y == 1, comp]), color='k', linestyle='dashed', linewidth=1)
        plt.title('PCA Component {}'.format(comp + 1))
        plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(list(range(1, len(pca.explained_variance_ratio_) + 1)), np.cumsum(pca.explained_variance_ratio_), '-o')
    plt.title('Cumulative % of Variance Explained by Components')
    plt.xlabel("No. of Components")
    plt.ylabel("Cumulative % of Variance")
    plt.savefig(IMAGE_DIR + 'pca_evaluation_{}'.format(dataset_name))
    plt.show()




    ##########################################
    # ICA feature transformation
    ##########################################

    from scipy.stats import kurtosis, skew

    plt.figure(figsize=(18, 4))

    avg_kurt = []
    for component in range(1, 18):
        # ICA feature transformation
        ica = FastICA(n_components=component, random_state=10)
        X_r = ica.fit_transform(data_x)
        X_ica = X_r

        avg_kurt.append(np.mean(np.apply_along_axis(kurtosis, 0, X_ica)))

    ica = FastICA(n_components=5, random_state=10)
    X_r = ica.fit_transform(data_x)
    X_ica = X_r
    kurtosis = np.apply_along_axis(kurtosis, 0, X_ica)

    colors = ["b","g","r","c","m","y","k"]
    lw = 2
    plt.subplot(1, 4, 1)

    for color, i in zip(colors, [0, 1]):
        plt.scatter(X_r[data_y == i, 0], X_r[data_y == i, 1], color=color, alpha=.8, lw=lw,
                    label='Label {}'.format(i))
    plt.xlabel("Transformed ICA Component 1")
    plt.ylabel("Transformed ICA Component 2")

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('ICA of Obesity dataset')

    plt.subplot(1, 4, 2)
    plt.title('Choosing number of components')
    plt.xlabel('No. of components')
    plt.ylabel('Average Kurtosis across IC')
    plt.plot(list(range(1, len(avg_kurt) + 1)), avg_kurt, '-o')
    plt.xticks(list(range(1, len(avg_kurt) + 1)))

    # ICA feature transformation
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    lw = 2

    plt.subplot(1, 4, 3)

    feature = 0
    plt.hist(X_r[data_y == 0, feature], density=True, alpha=0.5, color=colors[0], bins=30, label='Label 0')
    plt.axvline(np.median(X_r[data_y == 0, feature]), color='k', linestyle='dashed', linewidth=1)
    plt.hist(X_r[data_y == 1, feature], density=True, alpha=0.5, color=colors[1], bins=30, label='Label 1')
    plt.axvline(np.median(X_r[data_y == 1, feature]), color='k', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.title('ICA Component 1')

    import numpy as np

    plt.subplot(1, 4, 4)
    plt.title('Kurtosis Distribution (n=16)')
    plt.scatter(range(1, len(kurtosis) + 1), kurtosis, c='g')
    plt.ylabel('Kurtosis Value')
    plt.xlabel('Component No.')
    plt.xticks(list(range(1, len(kurtosis) + 1)))

    plt.savefig(IMAGE_DIR + 'ica_evaluation_{}'.format(dataset_name))
    plt.show()





    ##########################################
    # Random Projection feature transformation
    ##########################################

    states = [10, 13, 16, 19]
    plt.figure(figsize=(18, 4))

    for i, state in enumerate(states):

        rca = GaussianRandomProjection(n_components=17, random_state=state)
        X_r = rca.fit_transform(data_x)
        X_rca = X_r

        plt.subplot(1, 4, i + 1)
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        lw = 2

        for color, cp in zip(colors, [0, 1]):
            plt.scatter(X_r[data_y == cp, 0], X_r[data_y == cp, 1], color=color, alpha=.8, lw=lw,
                        label='Label {}'.format(cp))
        plt.xlabel("Transformed RCA Component 1")
        plt.ylabel("Transformed RCA Component 2")
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('Random Projection of Obesity dataset, Try No. {}'.format(i + 1))
    plt.savefig(IMAGE_DIR + 'rca_evaluation_{}'.format(dataset_name))
    plt.show()


    ##########################################
    # Univariate feature selection (K best)
    ##########################################

    from sklearn.feature_selection import chi2

    X_new = SelectKBest(chi2, k=4).fit_transform(data_x, data_y)
    X_fs = X_new

    plt.figure()
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    lw = 2

    for color, i in zip(colors, [0, 1]):
        plt.scatter(X_new[data_y == i, 1], X_new[data_y == i, 2], color=color, alpha=.8, lw=lw,
                    label='Label {}'.format(i))
    plt.xlabel("Transformed KBest Component 1")
    plt.ylabel("Transformed KBest Component 2")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Chi square feature selection of Obesity dataset')
    plt.savefig(IMAGE_DIR + 'kbest_evaluation_{}'.format(dataset_name))
    plt.show()

    ##########################################
    # k-mean
    ##########################################

    # PCA
    pca = PCA(n_components=3, random_state=10)
    X_r = pca.fit_transform(data_x)
    X_pca = X_r
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    kmean(X_pca, data_y, range_n_clusters, rounds, raw_data=False, label='pca')

    # ICA
    ica = FastICA(n_components=5, random_state=10)
    X_r = ica.fit_transform(data_x)
    X_ica = X_r
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    kmean(X_ica, data_y, range_n_clusters, rounds, raw_data=False, label='ica')

    # RCA
    rca = GaussianRandomProjection(n_components=4, random_state=10)
    X_r = rca.fit_transform(data_x)
    X_rca = X_r
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    kmean(X_rca, data_y, range_n_clusters, rounds, raw_data=False, label='rca')

    # KBest
    X_new = SelectKBest(chi2, k=4).fit_transform(data_x, data_y)
    X_fs = X_new
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(2, 0), (2, 1), (2, 3)]
    kmean(X_fs, data_y, range_n_clusters, rounds, raw_data=False, label='kbest')

    ##########################################
    # EM
    ##########################################

    # PCA
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_pca, data_y, range_n_clusters, rounds, raw_data=False, label='pca')

    # ICA
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_ica, data_y, range_n_clusters, rounds, raw_data=False, label='ica')

    # RCA
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(0, 1), (0, 2), (1, 2)]
    em(X_rca, data_y, range_n_clusters, rounds, raw_data=False, label='rca')

    # KBest
    # range_n_clusters = [2]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rounds = [(2, 0), (2, 1), (2, 3)]
    em(X_fs, data_y, range_n_clusters, rounds, raw_data=False, label='kbest')

























    #
    # # range_n_clusters = [2, 4, 6]
    # range_n_clusters = [4]
    #
    # for n_clusters in range_n_clusters:
    #     # Create a subplot with 1 row and 2 columns
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     fig.set_size_inches(18, 7)
    #
    #     # The 1st subplot is the silhouette plot
    #     # The silhouette coefficient can range from -1, 1 but in this example all
    #     # lie within [-0.1, 1]
    #     ax1.set_xlim([-0.1, 1])
    #     # The (n_clusters+1)*10 is for inserting blank space between silhouette
    #     # plots of individual clusters, to demarcate them clearly.
    #     ax1.set_ylim([0, len(X_pca) + (n_clusters + 1) * 10])
    #
    #     # Initialize the clusterer with n_clusters value and a random generator
    #     # seed of 10 for reproducibility.
    #     clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_pca)
    #     cluster_labels = clusterer.labels_
    #     print("NMI score: %.6f" % normalized_mutual_info_score(data_y, cluster_labels))
    #
    #     # The silhouette_score gives the average value for all the samples.
    #     # This gives a perspective into the density and separation of the formed
    #     # clusters
    #     silhouette_avg = silhouette_score(X_pca, cluster_labels)
    #     print("For n_clusters =", n_clusters,
    #           "The average silhouette_score is :", silhouette_avg)
    #
    #     # Compute the silhouette scores for each sample
    #     sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)
    #
    #     y_lower = 10
    #     for i in range(n_clusters):
    #         # Aggregate the silhouette scores for samples belonging to
    #         # cluster i, and sort them
    #         ith_cluster_silhouette_values = \
    #             sample_silhouette_values[cluster_labels == i]
    #
    #         ith_cluster_silhouette_values.sort()
    #
    #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #         y_upper = y_lower + size_cluster_i
    #
    #         cmap = cm.get_cmap("Spectral")
    #         color = cmap((float(i) + 0.5) / n_clusters)
    #
    #         ax1.fill_betweenx(np.arange(y_lower, y_upper),
    #                           0, ith_cluster_silhouette_values,
    #                           facecolor=color, edgecolor=color, alpha=0.7)
    #
    #         # Label the silhouette plots with their cluster numbers at the middle
    #         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    #
    #         # Compute the new y_lower for next plot
    #         y_lower = y_upper + 10  # 10 for the 0 samples
    #
    #     ax1.set_title("The silhouette plot for the various clusters.")
    #     ax1.set_xlabel("The silhouette coefficient values")
    #     ax1.set_ylabel("Cluster label")
    #
    #     # The vertical line for average silhouette score of all the values
    #     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    #
    #     ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #
    #     # 2nd Plot showing the actual clusters formed
    #     cmap = cm.get_cmap("Spectral")
    #     colors = cmap((cluster_labels.astype(float) + 0.5) / n_clusters)
    #
    #     ax2.scatter( X_pca[:, 0], X_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #                 c=colors, edgecolor='k')
    #
    #     # Labeling the clusters
    #     centers = clusterer.cluster_centers_
    #
    #     # Draw white circles at cluster centers
    #     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #                 c="white", alpha=1, s=200, edgecolor='k')
    #
    #     for i, c in enumerate(centers):
    #         ax2.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
    #                     s=50, edgecolor='k')
    #
    #     ax2.set_title("The visualization of the clustered data.")
    #     ax2.set_xlabel("Feature space for the 1st feature")
    #     ax2.set_ylabel("Feature space for the 2nd feature")
    #
    #     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
    #                   "with n_clusters = %d" % n_clusters),
    #                  fontsize=14, fontweight='bold')
    #
    #     plt.show()
    #
    #     # rounds = [(0, 10), (1, 6), (6, 8), (7, 10)]
    #     # rounds = [(3, 2), (1, 6), (0, 7), (13, 12)]
    #     # rounds = [(3, 2), (3, 1), (2, 1), (3, 6)]
    #     rounds = [(0, 1), (0, 2), (0, 3), (1, 2)]
    #     # rounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    #
    #     plt.figure(figsize=(15, 4))
    #
    #     for i, selected in enumerate(rounds):
    #         plt.subplot(1, 4, i + 1)
    #         plt.scatter(X_pca[:, selected[0]], X_pca[:, selected[1]], marker='.', s=30, lw=0, alpha=0.7,
    #                     c=colors, edgecolor='k')
    #
    #         # Labeling the clusters
    #         centers = clusterer.cluster_centers_
    #
    #         # Draw white circles at cluster centers
    #         plt.scatter(centers[:, selected[0]], centers[:, selected[1]], marker='o',
    #                     c="white", alpha=1, s=200, edgecolor='k')
    #
    #         for i, c in enumerate(centers):
    #             plt.scatter(c[selected[0]], c[selected[1]], marker='$%d$' % i, alpha=1,
    #                         s=50, edgecolor='k')
    #
    #         plt.title("{},{}".format(selected[0], selected[1]))
    #     plt.show()

    # #################################################
    # #Univariate feature selection (K best)
    #
    # from sklearn.feature_selection import chi2
    # from sklearn.feature_selection import mutual_info_classif
    #
    # X_new = SelectKBest(chi2, k=5).fit_transform(data_x, data_y)
    # X_fs = X_new
    #
    # plt.figure()
    # colors = ["b","g","r","c","m","y","k"]
    # lw = 2
    #
    # for color, i in zip(colors, [3,2]):
    #     plt.scatter(X_new[data_y == i, 4], X_new[data_y == i, 0], color=color, alpha=.8, lw=lw, label=i)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('Chi square feature selection of Wine Quality dataset')
    # plt.show()

    # #################################################
    # #Rerun clustering on transformed features
    # range_n_clusters = [2,4,6,8]
    # X_test=pd.DataFrame(X_pca)
    # for n_clusters in range_n_clusters:
    #     fig = plt.gcf()
    #     fig.set_size_inches(7, 7)
    #     ax = fig.add_subplot(111)
    #
    #     clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_test)
    #     cluster_labels = clusterer.labels_
    #
    #     silhouette_avg = silhouette_score(X_test, cluster_labels)
    #     print("For n_clusters =", n_clusters,
    #           "The average silhouette_score is :", silhouette_avg)
    #     print("The NMI score is: %.6f" % normalized_mutual_info_score(y, cluster_labels))
    #
    #     colors = plt.cm.Spectral(cluster_labels.astype(float) / n_clusters)
    #     ax.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #                 c=colors, edgecolor='k')
    #
    #     centers = clusterer.cluster_centers_
    #
    #     ax.scatter(centers[:, 0], centers[:, 1], marker='o',
    #                 c="white", alpha=1, s=200, edgecolor='k')
    #
    #     for i, c in enumerate(centers):
    #         ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
    #                     s=50, edgecolor='k')
    #
    #     ax.set_title("The visualization of the clustered data.")
    #     ax.set_xlabel("Feature space for the 1st feature")
    #     ax.set_ylabel("Feature space for the 2nd feature")
    #
    #     plt.suptitle(("KMeans clustering using PCA feature transformation "
    #                   "with n_clusters = %d" % n_clusters),
    #                  fontsize=14, fontweight='bold')
    #
    #     plt.show()
    #
    # X_test=pd.DataFrame(X_fs)
    # for n_clusters in range_n_clusters:
    #     fig = plt.gcf()
    #     fig.set_size_inches(7, 7)
    #     ax = fig.add_subplot(111)
    #
    #     clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_test)
    #     cluster_labels = clusterer.predict(X_test)
    #     print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))
    #
    #     colors = plt.cm.Spectral(cluster_labels.astype(float) / n_clusters)
    #     plt.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #                 c=colors, edgecolor='k')
    #
    #     centers = clusterer.means_
    #
    #     plt.scatter(centers[:, 0], centers[:, 1], marker='o',
    #                 c="white", alpha=1, s=200, edgecolor='k')
    #
    #     for i, c in enumerate(centers):
    #         ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
    #                     s=50, edgecolor='k')
    #
    #     ax.set_title("The visualization of the clustered data.")
    #     ax.set_xlabel("Feature space for the 1st feature")
    #     ax.set_ylabel("Feature space for the 2nd feature")
    #     plt.suptitle(("Clusters plot for EM clustering on PCA data "
    #                   "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    #
    #     plt.show()












# # Problem 1 - Four Peaks
# length = 30
# four_fitness = FourPeaks(t_pct=0.2)
# problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
# problem.mimic_speed = True
# random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs
#
# kwargs = {"rhc_max_iters": 1000,
#
#           "sa_max_iters": 1000,
#           "sa_init_temp": 100,
#           "sa_exp_decay_rate": 0.02,
#           "sa_min_temp": 0.001,
#
#           "ga_max_iters": 300,
#           "ga_pop_size": 900,
#           "ga_keep_pct": 0.5,
#
#           "mimic_max_iters": 200,
#           "mimic_pop_size": 900,
#           "mimic_keep_pct": 0.5,
#
#           "plot_name": 'Four_Peaks',
#           "plot_ylabel": 'Fitness'}
#
# # Initialize lists of fitness curves and time curves
# rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
# rhc_times, sa_times, ga_times, mimic_times = [], [], [], []
#
# # Set an exponential decay schedule for SA
# exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                      exp_const=kwargs['sa_exp_decay_rate'],
#                      min_temp=kwargs['sa_min_temp'])
#
# # For multiple random runs
# for random_state in random_states:
#     # Run RHC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = random_hill_climb(problem,
#                                                        max_attempts=kwargs['rhc_max_iters'],
#                                                        max_iters=kwargs['rhc_max_iters'],
#                                                        curve=True, random_state=random_state)
#
#     rhc_fitness.append(fitness_curve)
#     rhc_times.append(time.time() - start_time)
#     print('\nRHC: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run SA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = simulated_annealing(problem,
#                                                          schedule=exp_decay,
#                                                          max_attempts=kwargs['sa_max_iters'],
#                                                          max_iters=kwargs['sa_max_iters'],
#                                                          curve=True, random_state=random_state)
#
#     sa_fitness.append(fitness_curve)
#     sa_times.append(time.time() - start_time)
#     print('SA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run GA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = genetic_alg(problem,
#                                                  pop_size=kwargs['ga_pop_size'],
#                                                  mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                                  max_attempts=kwargs['ga_max_iters'],
#                                                  max_iters=kwargs['ga_max_iters'],
#                                                  curve=True, random_state=random_state)
#
#     ga_fitness.append(fitness_curve)
#     ga_times.append(time.time() - start_time)
#     print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run MIMIC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = mimic(problem,
#                                            pop_size=kwargs['mimic_pop_size'],
#                                            keep_pct=kwargs['mimic_keep_pct'],
#                                            max_attempts=kwargs['mimic_max_iters'],
#                                            max_iters=kwargs['mimic_max_iters'],
#                                            curve=True, random_state=random_state)
#
#     mimic_fitness.append(fitness_curve)
#     mimic_times.append(time.time() - start_time)
#     print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#
# print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
# print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
# print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
# print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))
#
#
# # Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
# rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
# sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
# ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
# mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
# utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
# utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
# utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
# utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
#                             x_label='Iterations',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
# print('\n')
#
# pop_size_list = [100, 300, 500, 700, 900]
# ga_pop_fitness_list = []
# mimic_pop_fitness_list = []
#
# for random_state in random_states:
#     ga_pop_fitness = []
#     mimic_pop_fitness = []
#     for i in range(len(pop_size_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=pop_size_list[i],
#                                          mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_pop_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=pop_size_list[i],
#                                    keep_pct=kwargs['mimic_keep_pct'],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_pop_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_pop_fitness_list.append(ga_pop_fitness)
#     mimic_pop_fitness_list.append(mimic_pop_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# # plot = plt.plot(pop_size_list, y_mean, label='GA')
# # plot = plt.plot(x_axis, y_mean, label='MIMIC')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
#                             x_label='population size',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
# print('\n')
#
#
# keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
# ga_keep_fitness_list = []
# mimic_keep_fitness_list = []
#
# for random_state in random_states:
#     ga_keep_fitness = []
#     mimic_keep_fitness = []
#     for i in range(len(keep_pct_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=kwargs['ga_pop_size'],
#                                          mutation_prob=1.0 - keep_pct_list[i],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_keep_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=kwargs['mimic_pop_size'],
#                                    keep_pct=keep_pct_list[i],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_keep_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_keep_fitness_list.append(ga_keep_fitness)
#     mimic_keep_fitness_list.append(mimic_keep_fitness)
# print('\n')
#
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
#                             x_label='keep_pct',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))
#
# temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
# sa_decay_fitness_list = []
#
# for random_state in random_states:
#     sa_decay_fitness = []
#     for i in range(len(temp_decay_list)):
#         # Set an exponential decay schedule for SA
#         exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                              exp_const=temp_decay_list[i],
#                              min_temp=kwargs['sa_min_temp'])
#
#         _, best_fitness, _ = simulated_annealing(problem,
#                                                  schedule=exp_decay,
#                                                  # max_attempts=kwargs['sa_max_iters'],
#                                                  # max_iters=kwargs['sa_max_iters'],
#                                                  max_attempts=1000,
#                                                  max_iters=1000,
#                                                  curve=True, random_state=random_state)
#         sa_decay_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#     sa_decay_fitness_list.append(sa_decay_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
# utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
#                             x_label='temp_decay_rate',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))
#
# print('\n')
#
#
#
#
#
#
#
# # Problem 2 - N-Queens
#
#
#
#
# # Define Queens objective function and problem
# length = 100
# queen = Queens()
# problem = DiscreteOpt(length=length, fitness_fn=queen, maximize=True, max_val=2)
# problem.mimic_speed = True  # set fast MIMIC
#
#
# random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs
#
# kwargs = {"rhc_max_iters": 600,
#
#           "sa_max_iters": 600,
#           "sa_init_temp": 100,
#           "sa_exp_decay_rate": 0.02,
#           "sa_min_temp": 0.001,
#
#           "ga_max_iters": 600,
#           "ga_pop_size": 900,
#           "ga_keep_pct": 0.5,
#
#           "mimic_max_iters": 50,
#           "mimic_pop_size": 900,
#           "mimic_keep_pct": 0.5,
#
#           "plot_name": 'Queen',
#           "plot_ylabel": 'Fitness'}
#
# # Initialize lists of fitness curves and time curves
# rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
# rhc_times, sa_times, ga_times, mimic_times = [], [], [], []
#
# # Set an exponential decay schedule for SA
# exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                      exp_const=kwargs['sa_exp_decay_rate'],
#                      min_temp=kwargs['sa_min_temp'])
#
# # For multiple random runs
# for random_state in random_states:
#     # Run RHC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = random_hill_climb(problem,
#                                                        max_attempts=kwargs['rhc_max_iters'],
#                                                        max_iters=kwargs['rhc_max_iters'],
#                                                        curve=True, random_state=random_state)
#
#     rhc_fitness.append(fitness_curve)
#     rhc_times.append(time.time() - start_time)
#     print('\nRHC: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run SA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = simulated_annealing(problem,
#                                                          schedule=exp_decay,
#                                                          max_attempts=kwargs['sa_max_iters'],
#                                                          max_iters=kwargs['sa_max_iters'],
#                                                          curve=True, random_state=random_state)
#
#     sa_fitness.append(fitness_curve)
#     sa_times.append(time.time() - start_time)
#     print('SA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run GA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = genetic_alg(problem,
#                                                  pop_size=kwargs['ga_pop_size'],
#                                                  mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                                  max_attempts=kwargs['ga_max_iters'],
#                                                  max_iters=kwargs['ga_max_iters'],
#                                                  curve=True, random_state=random_state)
#
#     ga_fitness.append(fitness_curve)
#     ga_times.append(time.time() - start_time)
#     print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run MIMIC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = mimic(problem,
#                                            pop_size=kwargs['mimic_pop_size'],
#                                            keep_pct=kwargs['mimic_keep_pct'],
#                                            max_attempts=kwargs['mimic_max_iters'],
#                                            max_iters=kwargs['mimic_max_iters'],
#                                            curve=True, random_state=random_state)
#
#     mimic_fitness.append(fitness_curve)
#     mimic_times.append(time.time() - start_time)
#     print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#
# print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
# print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
# print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
# print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))
#
#
# # Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
# rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
# sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
# ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
# mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
# utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
# utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
# utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
# utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
#                             x_label='Iterations',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
# print('\n')
#
# pop_size_list = [100, 300, 500, 700, 900]
# ga_pop_fitness_list = []
# mimic_pop_fitness_list = []
#
# for random_state in random_states:
#     ga_pop_fitness = []
#     mimic_pop_fitness = []
#     for i in range(len(pop_size_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=pop_size_list[i],
#                                          mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_pop_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=pop_size_list[i],
#                                    keep_pct=kwargs['mimic_keep_pct'],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_pop_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_pop_fitness_list.append(ga_pop_fitness)
#     mimic_pop_fitness_list.append(mimic_pop_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# # plot = plt.plot(pop_size_list, y_mean, label='GA')
# # plot = plt.plot(x_axis, y_mean, label='MIMIC')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
#                             x_label='population size',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
# print('\n')
#
#
# keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
# ga_keep_fitness_list = []
# mimic_keep_fitness_list = []
#
# for random_state in random_states:
#     ga_keep_fitness = []
#     mimic_keep_fitness = []
#     for i in range(len(keep_pct_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=kwargs['ga_pop_size'],
#                                          mutation_prob=1.0 - keep_pct_list[i],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_keep_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=kwargs['mimic_pop_size'],
#                                    keep_pct=keep_pct_list[i],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_keep_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_keep_fitness_list.append(ga_keep_fitness)
#     mimic_keep_fitness_list.append(mimic_keep_fitness)
# print('\n')
#
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
#                             x_label='keep_pct',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))
#
# temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
# sa_decay_fitness_list = []
#
# for random_state in random_states:
#     sa_decay_fitness = []
#     for i in range(len(temp_decay_list)):
#         # Set an exponential decay schedule for SA
#         exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                              exp_const=temp_decay_list[i],
#                              min_temp=kwargs['sa_min_temp'])
#
#         _, best_fitness, _ = simulated_annealing(problem,
#                                                  schedule=exp_decay,
#                                                  # max_attempts=kwargs['sa_max_iters'],
#                                                  # max_iters=kwargs['sa_max_iters'],
#                                                  max_attempts=1000,
#                                                  max_iters=1000,
#                                                  curve=True, random_state=random_state)
#         sa_decay_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#     sa_decay_fitness_list.append(sa_decay_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
# utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
#                             x_label='temp_decay_rate',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))
#
# print('\n')
#
#
#
#
# # Problem 3 - Knapsack
#
# weights = [10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
#            10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
#            10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22]
# values = [1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
#           1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
#           1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8]
# max_weight_pct = 0.6
# n = len(weights)
# # Initialize fitness function object using coords_list
# fitness = Knapsack(weights, values, max_weight_pct)
# # Define optimization problem object
# problem = DiscreteOpt(length=n, fitness_fn=fitness, maximize=True)
# problem.mimic_speed = True  # set fast MIMIC
#
#
#
# random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs
#
# kwargs = {"rhc_max_iters": 600,
#
#           "sa_max_iters": 600,
#           "sa_init_temp": 100,
#           "sa_exp_decay_rate": 0.02,
#           "sa_min_temp": 0.001,
#
#           "ga_max_iters": 600,
#           "ga_pop_size": 900,
#           "ga_keep_pct": 0.5,
#
#           "mimic_max_iters": 100,
#           "mimic_pop_size": 900,
#           "mimic_keep_pct": 0.5,
#
#           "plot_name": 'Knapsack',
#           "plot_ylabel": 'Fitness'}
#
# # Initialize lists of fitness curves and time curves
# rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
# rhc_times, sa_times, ga_times, mimic_times = [], [], [], []
#
# # Set an exponential decay schedule for SA
# exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                      exp_const=kwargs['sa_exp_decay_rate'],
#                      min_temp=kwargs['sa_min_temp'])
#
# # For multiple random runs
# for random_state in random_states:
#     # Run RHC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = random_hill_climb(problem,
#                                                        max_attempts=kwargs['rhc_max_iters'],
#                                                        max_iters=kwargs['rhc_max_iters'],
#                                                        curve=True, random_state=random_state)
#
#     rhc_fitness.append(fitness_curve)
#     rhc_times.append(time.time() - start_time)
#     print('\nRHC: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run SA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = simulated_annealing(problem,
#                                                          schedule=exp_decay,
#                                                          max_attempts=kwargs['sa_max_iters'],
#                                                          max_iters=kwargs['sa_max_iters'],
#                                                          curve=True, random_state=random_state)
#
#     sa_fitness.append(fitness_curve)
#     sa_times.append(time.time() - start_time)
#     print('SA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run GA and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = genetic_alg(problem,
#                                                  pop_size=kwargs['ga_pop_size'],
#                                                  mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                                  max_attempts=kwargs['ga_max_iters'],
#                                                  max_iters=kwargs['ga_max_iters'],
#                                                  curve=True, random_state=random_state)
#
#     ga_fitness.append(fitness_curve)
#     ga_times.append(time.time() - start_time)
#     print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#     # Run MIMIC and get best state and objective found
#     start_time = time.time()
#     _, best_fitness, fitness_curve = mimic(problem,
#                                            pop_size=kwargs['mimic_pop_size'],
#                                            keep_pct=kwargs['mimic_keep_pct'],
#                                            max_attempts=kwargs['mimic_max_iters'],
#                                            max_iters=kwargs['mimic_max_iters'],
#                                            curve=True, random_state=random_state)
#
#     mimic_fitness.append(fitness_curve)
#     mimic_times.append(time.time() - start_time)
#     print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#
# print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
# print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
# print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
# print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))
#
#
# # Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
# rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
# sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
# ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
# mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
# utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
# utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
# utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
# utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
#                             x_label='Iterations',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
# print('\n')
#
# pop_size_list = [100, 300, 500, 700, 900]
# ga_pop_fitness_list = []
# mimic_pop_fitness_list = []
#
# for random_state in random_states:
#     ga_pop_fitness = []
#     mimic_pop_fitness = []
#     for i in range(len(pop_size_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=pop_size_list[i],
#                                          mutation_prob=1.0 - kwargs['ga_keep_pct'],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_pop_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=pop_size_list[i],
#                                    keep_pct=kwargs['mimic_keep_pct'],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_pop_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_pop_fitness_list.append(ga_pop_fitness)
#     mimic_pop_fitness_list.append(mimic_pop_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# # plot = plt.plot(pop_size_list, y_mean, label='GA')
# # plot = plt.plot(x_axis, y_mean, label='MIMIC')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
# utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
#                             x_label='population size',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
# print('\n')
#
#
# keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
# ga_keep_fitness_list = []
# mimic_keep_fitness_list = []
#
# for random_state in random_states:
#     ga_keep_fitness = []
#     mimic_keep_fitness = []
#     for i in range(len(keep_pct_list)):
#         # Run GA and get best state and objective found
#         _, best_fitness, _ = genetic_alg(problem,
#                                          pop_size=kwargs['ga_pop_size'],
#                                          mutation_prob=1.0 - keep_pct_list[i],
#                                          max_attempts=kwargs['ga_max_iters'],
#                                          max_iters=kwargs['ga_max_iters'],
#                                          curve=True, random_state=random_state)
#
#         ga_keep_fitness.append(best_fitness)
#         print('GA: best_objective = {:.3f}'.format(best_fitness))
#
#         # Run MIMIC and get best state and objective found
#         _, best_fitness, _ = mimic(problem,
#                                    pop_size=kwargs['mimic_pop_size'],
#                                    keep_pct=keep_pct_list[i],
#                                    max_attempts=kwargs['mimic_max_iters'],
#                                    max_iters=kwargs['mimic_max_iters'],
#                                    curve=True, random_state=random_state)
#
#         mimic_keep_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#
#     ga_keep_fitness_list.append(ga_keep_fitness)
#     mimic_keep_fitness_list.append(mimic_keep_fitness)
# print('\n')
#
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
# utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
# utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
#                             x_label='keep_pct',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))
#
# temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
# sa_decay_fitness_list = []
#
# for random_state in random_states:
#     sa_decay_fitness = []
#     for i in range(len(temp_decay_list)):
#         # Set an exponential decay schedule for SA
#         exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
#                              exp_const=temp_decay_list[i],
#                              min_temp=kwargs['sa_min_temp'])
#
#         _, best_fitness, _ = simulated_annealing(problem,
#                                                  schedule=exp_decay,
#                                                  # max_attempts=kwargs['sa_max_iters'],
#                                                  # max_iters=kwargs['sa_max_iters'],
#                                                  max_attempts=1000,
#                                                  max_iters=1000,
#                                                  curve=True, random_state=random_state)
#         sa_decay_fitness.append(best_fitness)
#         print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
#     sa_decay_fitness_list.append(sa_decay_fitness)
#
# # Plot objective curves, set title and labels
# plt.figure()
# utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
# utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
#                             x_label='temp_decay_rate',
#                             y_label=kwargs['plot_ylabel'])
#
# # Save figure
# plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))
#
# print('\n')
#
#
#
#
#
#
# # NN
#
# def process_dataset(dataset_name):
#     if dataset_name == "obesity":
#
#         data = pd.read_csv(os.path.join("datasets", "ObesityDataSet_raw_and_data_sinthetic.csv"))
#
#         data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
#         data['family_history_with_overweight'] = data['family_history_with_overweight'].map({'no': 0, 'yes': 1})
#         data['FAVC'] = data['FAVC'].map({'no': 0, 'yes': 1})
#         data['CAEC'] = data['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
#         data['SMOKE'] = data['SMOKE'].map({'no': 0, 'yes': 1})
#         data['SCC'] = data['SCC'].map({'no': 0, 'yes': 1})
#         data['CALC'] = data['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
#         data['MTRANS'] = data['MTRANS'].map({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3,
#                                              'Automobile': 4})
#         data['NObeyesdad'] = data['NObeyesdad'].map({'Insufficient_Weight': 0,
#                                                      'Normal_Weight': 1,
#                                                      'Overweight_Level_I': 2,
#                                                      'Overweight_Level_II': 3,
#                                                      'Obesity_Type_I': 4,
#                                                      'Obesity_Type_II': 5,
#                                                      'Obesity_Type_III': 6})
#
#     elif dataset_name == "online_shopping":
#
#         data = pd.read_csv(os.path.join("datasets", "online_shoppers_intention.csv"))
#
#         data['Month'] = data['Month'].map({'Feb': 2,
#                                            'Mar': 3,
#                                            'May': 5,
#                                            'June': 6,
#                                            'Jul': 7,
#                                            'Aug': 8,
#                                            'Sep': 9,
#                                            'Oct': 10,
#                                            'Nov': 11,
#                                            'Dec': 12})
#         data['VisitorType'] = data['VisitorType'].map({'Returning_Visitor': 0,
#                                                        'New_Visitor': 1,
#                                                        'Other': 2})
#         data['Weekend'] = data['Weekend'].astype(int)
#         data['Revenue'] = data['Revenue'].astype(int)
#
#     else:
#         data = []
#
#     return data
#
#
# def split_data(data, testing_raio=0.2, norm=False):
#     data_matrix = data.values
#
#     def scale(col, min, max):
#         range = col.max() - col.min()
#         a = (col - col.min()) / range
#         return a * (max - min) + min
#
#     if norm and data_matrix.shape[1] == 17:
#         data_matrix[:, 1] = scale(data_matrix[:, 1], 0, 5)
#         data_matrix[:, 2] = scale(data_matrix[:, 2], 0, 5)
#         data_matrix[:, 3] = scale(data_matrix[:, 3], 0, 5)
#
#     x = data_matrix[:, :-1]
#     y = data_matrix[:, -1]
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_raio, shuffle=True,
#                                                         random_state=42, stratify=y)
#
#     return x_train, x_test, y_train, y_test
#
#
# # import and process datasets
# dataset_name_list = ["online_shopping"]
# datas = []
# for dataset_name in dataset_name_list:
#     datas.append(process_dataset(dataset_name))
#
# kfold = KFold(n_splits=10, shuffle=True, random_state=1)
#
# for data_num, data in enumerate(datas):
#     dataset_name = dataset_name_list[data_num]
#     # Split data
#     if dataset_name == "obesity":
#         x_train, x_test, y_train, y_test = split_data(data, norm=True)
#     else:
#         x_train, x_test, y_train, y_test = split_data(data)
#
#     print("Training Set Shape: {}".format(x_train.shape))
#     print("Testing Set Shape: {}".format(x_test.shape))
#
#
#
#
# random_seeds = [1234 + 1 * i for i in range(1)]
#
# # iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])
# iterations = np.array([5, 10, 25, 50, 100, 200, 300])
# # iterations = np.array([5, 25])
#
#
# kwargs = {"random_seeds": random_seeds,
#           "rhc_max_iters": iterations,
#           "sa_max_iters": iterations,
#           "ga_max_iters": iterations,
#           "init_temp": 100,
#           "exp_decay_rate": 0.1,
#           "min_temp": 0.001,
#           "pop_size": 100,
#           "mutation_prob": 0.2,
#           }
#
# # Initialize algorithms, corresponding acronyms and max number of iterations
# algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
# acronyms = ['RHC', 'SA', 'GA']
# max_iters = ['rhc_max_iters', 'sa_max_iters', 'ga_max_iters']
#
# # Initialize lists of training curves, validation curves and training times curves
# train_curves, val_curves, train_time_curves = [], [], []
#
# # Define SA exponential decay schedule
# exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
#                      exp_const=kwargs['exp_decay_rate'],
#                      min_temp=kwargs['min_temp'])
#
# # Create one figure for training and validation losses, the second for training time
# plt.figure()
# train_val_figure = plt.gcf().number
# plt.figure()
# train_times_figure = plt.gcf().number
# marker = ['+', 'x', 'o']
# # For each of the optimization algorithms to test the Neural Network with
# for i, algorithm in enumerate(algorithms):
#     print('\nAlgorithm = {}'.format(algorithm))
#
#     # For multiple random runs
#     for random_seed in random_seeds:
#
#         # Initialize training losses, validation losses and training time lists for current random run
#         train_losses, val_losses, train_times = [], [], []
#
#         # Compute stratified k-fold
#         x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
#                                                                               test_size=0.2, shuffle=True,
#                                                                               random_state=random_seed,
#                                                                               stratify=y_train)
#         # For each max iterations to run for
#         for max_iter in kwargs[max_iters[i]]:
#             # Define Neural Network using current algorithm
#             nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
#                                algorithm=algorithm, max_iters=int(max_iter),
#                                bias=True, is_classifier=True, learning_rate=0.001,
#                                early_stopping=False, clip_max=1e10, schedule=exp_decay,
#                                pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
#                                max_attempts=int(max_iter), random_state=random_seed, curve=False)
#
#             # Train on current training fold and append training time
#             start_time = time.time()
#             nn.fit(x_train_fold, y_train_fold)
#             train_times.append(time.time() - start_time)
#
#             # Compute and append training and validation log losses
#             train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
#             val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             print('{} - train loss = {:.3f}, val loss = {:.3f}'.format(max_iter, train_loss, val_loss))
#
#         # Append curves for current random seed to corresponding lists of curves
#         train_curves.append(train_losses)
#         val_curves.append(val_losses)
#         train_time_curves.append(train_times)
#
#     # Plot training and validation figure for current algorithm
#     plt.figure(train_val_figure)
#     # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_curves), label='{} train'.format(acronyms[i]))
#     # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(val_curves), label='{} val'.format(acronyms[i]))
#     plt.plot(kwargs[max_iters[i]], np.mean(np.array(train_curves), axis=0), marker=marker[i], label='{} train'.format(acronyms[i]))
#     plt.plot(kwargs[max_iters[i]], np.mean(np.array(val_curves), axis=0), marker=marker[i], label='{} test'.format(acronyms[i]))
#
#     # Plot training time figure for current algorithm
#     plt.figure(train_times_figure)
#     # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_time_curves), label=acronyms[i])
#     plt.plot(kwargs[max_iters[i]], np.mean(np.array(train_time_curves), axis=0), marker=marker[i], label='{} test'.format(acronyms[i]))
#
# # Set title and labels to training and validation figure
# plt.figure(train_val_figure)
# utils.set_plot_title_labels(title='Neural Network - Loss vs. iterations',
#                             x_label='Iterations',
#                             y_label='Loss')
#
# # Save figure
# plt.savefig(IMAGE_DIR + 'nn_objective_vs_iterations')
#
# # Set title and labels to training time figure
# plt.figure(train_times_figure)
# utils.set_plot_title_labels(title='Neural Network - Time vs. iterations',
#                             x_label='Iterations',
#                             y_label='Time (seconds)')
#
# # Save figure
# plt.savefig(IMAGE_DIR + 'nn_time_vs_iterations')
#
#
#
#
#
#
#
#
#
# kwargs = {"random_seeds": random_seeds[0],
#           "max_iters": 200,
#           "init_temp": 100,
#           "exp_decay_rate": 0.1,
#           "min_temp": 0.001,
#           "pop_size": 100,
#           "mutation_prob": 0.2,
#           }
#
# # Define SA exponential decay schedule
# exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
#                      exp_const=kwargs['exp_decay_rate'],
#                      min_temp=kwargs['min_temp'])
#
# # Define Neural Network using RHC for weights optimization
# rhc_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
#                        algorithm='random_hill_climb', max_iters=kwargs['max_iters'],
#                        bias=True, is_classifier=True, learning_rate=0.001,
#                        early_stopping=False, clip_max=1e10,
#                        max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
#
# # Define Neural Network using SA for weights optimization
# sa_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
#                       algorithm='simulated_annealing', max_iters=kwargs['max_iters'],
#                       bias=True, is_classifier=True, learning_rate=0.001,
#                       early_stopping=False, clip_max=1e10, schedule=exp_decay,
#                       max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
#
# # Define Neural Network using GA for weights optimization
# ga_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
#                       algorithm='genetic_alg', max_iters=kwargs['max_iters'],
#                       bias=True, is_classifier=True, learning_rate=0.001,
#                       early_stopping=False, clip_max=1e10,
#                       pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
#                       max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
#
# # Fit each of the Neural Networks using the different optimization algorithms
# # mimic_nn.fit(x_train, y_train)
# rhc_nn.fit(x_train, y_train)
# sa_nn.fit(x_train, y_train)
# ga_nn.fit(x_train, y_train)
#
# # https: // towardsdatascience.com / metrics - to - evaluate - your - machine - learning - algorithm - f10ba6e38234
# # right now outputs F1 score --> 2*precision*recall/(precision+recall)
#
# # Print classification reports for all of the optimization algorithms
# # print('MIMIC test classification report = \n {}'.format(classification_report(y_test, mimic_nn.predict(x_test))))
# print('RHC test classification report = \n {}'.format(classification_report(y_test, rhc_nn.predict(x_test))))
# print('SA test classification report = \n {}'.format(classification_report(y_test, sa_nn.predict(x_test))))
# print('GA test classification report = \n {}'.format(classification_report(y_test, ga_nn.predict(x_test))))







