# Complete version for Crop dataset, showed better result than K-Means
# 08/02/2019
import csv
import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
from collections import Counter
import math
import numpy as np


# Method to read the data.
def read_data(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        df = pd.DataFrame(list(reader))

        # The first row became headers
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header

        header_list = list(df)
        numerical_data = []
        for i in range(len(header_list)):
            numerical_data.append(pd.to_numeric(df[header_list[i]]))

        numerical_df = pd.DataFrame({header_list[i]: numerical_data[i] for i in range(len(header_list))})

        numerical_df.reset_index(inplace=True)

        numerical_df = numerical_df.drop(['index'], axis=1)

        target = numerical_df['target']
        numerical_df = numerical_df.drop(['target'], axis=1)

        return [numerical_df, target]

# Method for checking K-means clustering method, correct rate: around 52%
def error_rate_check(cluster_result):
    result = 0
    for i in range(24):
        counter_result = Counter(cluster_result[0 + 300 * i:299 + 300 * i]).most_common(1)
        result = result + list(counter_result[0])[1]

    print(result / 7200)


# New method used to improve the performance of K-Means
def hierarchical_based_Kmeans(data_set, starting_num,target_num):
    # Step 1: Use kmeans cluster to calculate the initial cluster, and get the labels and centroids
    # it uses random seed.
    # the result would be the cluster result and centroid list.
    cluster = KMeans(n_clusters=starting_num,random_state=0).fit(data_set)
    cluster_result = cluster.labels_
    centroid_list = cluster.cluster_centers_
    num_clusters = starting_num

    while True:
        score_list = []
        for i in range(num_clusters):
            result = sub_cluster_generation(data_set,i,cluster_result,centroid_list[i])
            score_list.append(result)


        cluster_to_divide = score_list.index(max(score_list))


        sub_centroids_list = cluster_divide(data_set,cluster_to_divide,cluster_result)

        centroid_list[cluster_to_divide] = sub_centroids_list[0]
        lst = list(centroid_list)
        lst.append(sub_centroids_list[1])
        centroid_list = np.asarray(lst)


        cluster = KMeans(n_clusters=len(centroid_list), random_state = 0, init = centroid_list, n_init = 1).fit(data_set)
        cluster_result = cluster.labels_
        centroid_list = cluster.cluster_centers_
        num_clusters = len(centroid_list)

        print(num_clusters)

        if num_clusters == target_num:
            break
    return [cluster_result,centroid_list, num_clusters]


# Method used to generate sub clusters.
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


        cluster_result = AgglomerativeClustering(n_clusters=2, linkage = 'single').fit(temp_df.transpose())

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

# Method used to divide the clusters.
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

    cluster_result = AgglomerativeClustering(n_clusters=2, linkage = 'single').fit(temp_df.transpose())

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

[data_set, target] = read_data('../Dataset/Crop_TEST.csv')
# [cluster_result,cluster_centroids,num_cluster] = hierarchical_based_Kmeans(data_set,2,24)
# #
# error_rate_check(cluster_result)
#
# cluster = KMeans(n_clusters=24).fit(data_set)
# error_rate_check(cluster.labels_)

