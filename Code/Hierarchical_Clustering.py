import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabaz_score

from sklearn import datasets
def read_data(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        cluster_result = list(reader)

        df = pd.DataFrame(cluster_result)

        # The first row became headers
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        # The first column became index
        df = df.set_index('index')

        header_list = list(df)

        numerical_data = []
        for i in range(len(header_list)):
            numerical_data.append(pd.to_numeric(df[header_list[i]]))

        return pd.DataFrame({header_list[i]: numerical_data[i] for i in range(len(header_list))})

def clusters_visualization(data_set, cluster_result, cluster_num):
    gp_col = 'Cluster'
    header_list = list(data_set)
    temp_set = data_set.copy()
    temp_set[gp_col] = cluster_result

    df_mean = temp_set.groupby(gp_col)[header_list].mean()
    df_mean = df_mean.transpose()

    fig_title = 'Hierarchical Clustering - ' + str(cluster_num) + ' clusters'
    file_name = 'Hierarchical Clustering - ' + str(cluster_num) + ' clusters'
    fig = df_mean.plot(grid = True, title = fig_title)
    fig.set_xlabel('Date')
    fig.set_ylabel('Percentage')
    #fig.set_xticklabels(df_mean.index.tolist())
    fig = fig.get_figure()
    file_path = '../Graph/'
    fig.savefig(file_path + file_name)
    plt.show()

def single_cluster_visualization(data_set, cluster_result, cluster_num):
    temp_set = data_set.copy()
    temp_set = temp_set.transpose()
    header_list = list(temp_set)
    temp_df = pd.DataFrame()
    for i in range(len(header_list)):
        if cluster_result[i] == cluster_num:
            temp_df[header_list[i]] = temp_set[header_list[i]]

    fig_title = 'Hierarchical Clustering - Cluster' + str(cluster_num)
    file_name = 'Hierarchical Clustering - Cluster' + str(cluster_num)
    fig = temp_df.plot(grid = True, title = fig_title)
    fig.set_xlabel('Date')
    fig.set_ylabel('Percentage')
    fig = fig.get_figure()
    file_path = '../Graph/'
    fig.savefig(file_path + file_name)
    plt.show()

def silhouette_score_evaluation(data_set,cluster_result):
    return silhouette_score(data_set, cluster_result, metric= 'euclidean')

def calinski_harabaz_score_evaluation(data_set,cluster_result):
    return calinski_harabaz_score(data_set,cluster_result)

def davies_bouldin_score_evaluation(data_set,cluster_result):
    return davies_bouldin_score(data_set, cluster_result)

data_set = read_data('../Dataset/S&P500_Data_Norm1.csv')
# cluster_model = AgglomerativeClustering(affinity='euclidean',linkage='ward',n_clusters=2).fit(data_set)
# list = []
for i in range(4,26):
    print(i)
    result = 0
    for j in range(20):
        cluster_model = AgglomerativeClustering(affinity='euclidean',linkage='ward',n_clusters=i).fit(data_set)
        result += silhouette_score_evaluation(data_set,cluster_model.labels_)
    average = result / 20
    print(average)
# num_cluster = 10
# cluster_model = AgglomerativeClustering(affinity='euclidean',linkage='average',n_clusters=num_cluster).fit(data_set)
#
# clusters_visualization(data_set,cluster_model.labels_,num_cluster)
# for i in range (num_cluster):
#     single_cluster_visualization(data_set,cluster_model.labels_,i)