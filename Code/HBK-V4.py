# first step is to use kmeans, then use hierarchical ward method
# to split each sub cluster into two clusters then calculate the sum of squard error
# previous - current, the higher, the more urgent to be splited

# 08/02/2019 - Complete version for the new algorithm. Showed better result.
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabaz_score
import math
import numpy as np
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
def pattern_visualization(data_set, cluster_result, cluster_num):
    gp_col = 'Cluster'
    header_list = list(data_set)
    temp_set = data_set.copy()
    temp_set[gp_col] = cluster_result

    df_mean = temp_set.groupby(gp_col)[header_list].mean()
    df_mean = df_mean.transpose()

    fig_title = 'Hierarchical_Based_KMeans algorithm - ' + str(cluster_num) + ' clusters'
    file_name = 'Hierarchical_Based_KMeans algorithm - ' + str(cluster_num) + ' clusters'
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

    fig_title = 'Hierarchical_Based_KMeans algorithm - Cluster' + str(cluster_num)
    file_name = 'Hierarchical_Based_KMeans algorithm - Cluster' + str(cluster_num)
    fig = temp_df.plot(grid = True, title = fig_title)
    fig.set_xlabel('Date')
    fig.set_ylabel('Percentage')
    fig = fig.get_figure()
    file_path = '../Graph/'
    fig.savefig(file_path + file_name)
    plt.show()
def hierarchical_based_Kmeans(data_set, starting_num,target_num):
    # Step 1: Use kmeans cluster to calculate the initial cluster, and get the labels and centroids
    # it uses random seed.
    # the result would be the cluster result and centroid list.
    cluster = KMeans(n_clusters=starting_num).fit(data_set)
    cluster_result = cluster.labels_
    centroid_list = cluster.cluster_centers_
    num_clusters = starting_num

    score_list = []
    cluster_to_divide = 0

    while True:

        if num_clusters == starting_num:
            for i in range(num_clusters):
                result = sub_cluster_generation(data_set,i,cluster_result,centroid_list[i])
                score_list.append(result)
        else:
            result = sub_cluster_generation(data_set,cluster_to_divide,cluster_result,centroid_list[cluster_to_divide])
            score_list[cluster_to_divide] = result

            result = sub_cluster_generation(data_set,num_clusters - 1,cluster_result,centroid_list[num_clusters - 1])
            score_list.append(result)

        cluster_to_divide = score_list.index(max(score_list))

        sub_centroids_list = cluster_divide(data_set,cluster_to_divide,cluster_result)

        centroid_list[cluster_to_divide] = sub_centroids_list[0]
        lst = list(centroid_list)
        lst.append(sub_centroids_list[1])
        centroid_list = np.asarray(lst)

        cluster = KMeans(n_clusters=len(centroid_list), init = centroid_list, n_init = 1).fit(data_set)
        cluster_result = cluster.labels_
        centroid_list = cluster.cluster_centers_
        num_clusters = len(centroid_list)

        if num_clusters == target_num:
            break
    return [cluster_result,centroid_list, num_clusters]

def sub_cluster_generation(data_set, cluster_num,cluster_result,centroid):
    # copy the original set and then transpose
    temp_set = data_set.copy()
    temp_set = temp_set.transpose()
    header_list = list(temp_set)
    temp_df = pd.DataFrame()

    # add each time series to the cluster where it belongs to.
    for i in range(len(header_list)):
        if cluster_result[i] == cluster_num:
            temp_df[header_list[i]] = temp_set[header_list[i]]

    # send the sub_cluster to recheck, calculate the error score and then send it back.
    error_score = sub_cluster_recheck(temp_df,centroid)
    return error_score

def sub_cluster_recheck(data_set, centroid):
    headers = list(data_set)
    num = len(headers)
    temp_df = data_set.copy()


    if num == 1:
        return 0
    elif num == 2:
        sum_of_squared_error = 0
        for i in range(num):
            for j in range(len(centroid)):
                # calculate the sum of squared error
                sum_of_squared_error += math.pow(centroid[j] - temp_df[headers[i]][j],2)

        return sum_of_squared_error
    else:
        # if there are more than two stocks in one cluster, compare each stock with the
        # centroid
        sum_of_squared_error = 0

        for i in range(num):
            for j in range(len(centroid)):
                # calculate the sum of squared error
                sum_of_squared_error += math.pow(centroid[j] - temp_df[headers[i]][j], 2)


        cluster_result = AgglomerativeClustering(n_clusters=2, linkage = 'average').fit(temp_df.transpose())

        cluster1 = pd.DataFrame()
        cluster2 = pd.DataFrame()

        header_list = list(temp_df)


        for i in range(len(header_list)):
            if cluster_result.labels_[i] == 0:
                cluster1[header_list[i]] = temp_df[header_list[i]]
            else:
                cluster2[header_list[i]] = temp_df[header_list[i]]


        cluster1_headers = list(cluster1)
        cluster2_headers = list(cluster2)
        cluster1_num = len(cluster1_headers)
        cluster2_num = len(cluster2_headers)
        cluster1_centroid = cluster1.mean(axis = 1)
        cluster2_centroid = cluster2.mean(axis=1)

        new_sum_of_squared_error = 0
        for i in range(cluster1_num):
            for j in range(len(cluster1_centroid)):
                new_sum_of_squared_error = new_sum_of_squared_error + math.pow(cluster1_centroid[j] - cluster1[cluster1_headers[i]][j], 2)

        for i in range(cluster2_num):
            for j in range(len(cluster2_centroid)):
                new_sum_of_squared_error = new_sum_of_squared_error + math.pow(cluster2_centroid[j] - cluster2[cluster2_headers[i]][j], 2)

        return (sum_of_squared_error - new_sum_of_squared_error)


def cluster_divide(data_set, cluster_num,cluster_result):
    # copy the original set and then transpose
    temp_set = data_set.copy()
    temp_set = temp_set.transpose()
    header_list = list(temp_set)
    temp_df = pd.DataFrame()

    # add each time series to the cluster where it belongs to.
    for i in range(len(header_list)):
        if cluster_result[i] == cluster_num:
            temp_df[header_list[i]] = temp_set[header_list[i]]

    cluster_result = AgglomerativeClustering(n_clusters=2, linkage = 'average').fit(temp_df.transpose())

    cluster1 = pd.DataFrame()
    cluster2 = pd.DataFrame()

    header_list = list(temp_df)
    for i in range(len(header_list)):
        if cluster_result.labels_[i] == 0:
            cluster1[header_list[i]] = temp_df[header_list[i]]
        else:
            cluster2[header_list[i]] = temp_df[header_list[i]]

    cluster1_centroid = cluster1.mean(axis=1)
    cluster2_centroid = cluster2.mean(axis=1)

    sub_centroid_list = []
    sub_centroid_list.append(cluster1_centroid)
    sub_centroid_list.append(cluster2_centroid)
    return sub_centroid_list

def silhouette_score_evaluation(data_set,cluster_result):
    score = silhouette_score(data_set, cluster_result, metric= 'euclidean')
    return score

def calinski_harabaz_score_evaluation(data_set,cluster_result):
    return calinski_harabaz_score(data_set,cluster_result)

def davies_bouldin_score_evaluation(data_set,cluster_result):
    return davies_bouldin_score(data_set, cluster_result)

data_set = read_data('../Dataset/FTSE100_Data_Norm1.csv')
target_num = 10
start_num = 2
for i in range(9,10):
    score = 0
    for j in range(5):
        [cluster_result,cluster_centroids,num_cluster] = hierarchical_based_Kmeans(data_set,i,target_num)
        score += silhouette_score_evaluation(data_set, cluster_result)
        # print(silhouette_score_evaluation(data_set, cluster_result))
    print(score / 5)

# clusters_visualization(data_set,cluster_result,target_num)
# for i in range (target_num):
#     single_cluster_visualization(data_set,cluster_result,i)
#
# Kmeans_result = KMeans(n_clusters = target_num).fit(data_set)
# print((silhouette_score_evaluation(data_set,Kmeans_result.labels_)))
