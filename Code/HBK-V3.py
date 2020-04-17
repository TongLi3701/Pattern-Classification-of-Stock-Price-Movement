# first step is to use kmeans, then use hierarchical ward method
# to split each sub cluster into two clusters then calculate the sum of squard error
# previous - current, the higher, the more urgent to be splited

# 08/02/2019 - added one more step, set the upper bound
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
def clusters_visualization(data_set, cluster_result, cluster_num):
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
    # fig.savefig(file_name)
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
    # fig.savefig(file_name)
    plt.show()
def hierarchical_based_Kmeans(data_set, starting_num,target_num, upper_bound):
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
        #
        #
        cluster = KMeans(n_clusters=len(centroid_list), random_state = 0, init = centroid_list, n_init = 1).fit(data_set)
        cluster_result = cluster.labels_
        centroid_list = cluster.cluster_centers_
        num_clusters = len(centroid_list)

        print(num_clusters)

        if num_clusters == upper_bound:
            break


    # While loop used to recheck each subcluster, make sure that the first K-Means result is good.
    # while True:
    #     score_table = []
    #     for i in range(num_clusters):
    #         for j in range(num_clusters):
    #             score = pair_clusters_recheck(data_set,cluster_result,i,j)
    while True:
        min_score = float('inf')
        cluster_num1 = 0
        cluster_num2 = 0
        for i in range(num_clusters):
            for j in range(num_clusters):
                if (i != j):
                    score = pair_clusters_recheck(data_set, cluster_result, i, j)
                    if (score < min_score):
                        min_score = score
                        cluster_num1 = i
                        cluster_num2 = j

        new_centroid_list = pair_clusters_converge(data_set,cluster_result,centroid_list,cluster_num1,cluster_num2)

        cluster = KMeans(n_clusters=len(new_centroid_list), random_state=0, init=new_centroid_list, n_init=1).fit(data_set)
        cluster_result = cluster.labels_
        centroid_list = cluster.cluster_centers_
        num_clusters = len(centroid_list)

        print(num_clusters)

        if num_clusters == target_num:
            break

    return [cluster_result,centroid_list, num_clusters]

def pair_clusters_converge(data_set, cluster_result,centroid_list,num1, num2):
    temp_set = data_set.copy()
    temp_set = temp_set.transpose()
    header_list = list(temp_set)
    combined_cluster = pd.DataFrame()

    for i in range(len(header_list)):
        if(cluster_result[i]) == num1 or (cluster_result[i]) == num2:
            combined_cluster[header_list[i]] = temp_set[header_list[i]]

    new_centroid = combined_cluster.mean(axis = 1)

    new_centroid_list = []

    for i in range(len(centroid_list)):
        if i != num1 and i != num2:
            new_centroid_list.append(centroid_list[i])

    new_centroid_list.append(new_centroid)

    return np.asarray((new_centroid_list))



def pair_clusters_recheck(data_set,cluster_result,num1,num2):
    temp_set = data_set.copy()
    temp_set = temp_set.transpose()
    header_list = list(temp_set)
    # Create two empty clusters
    cluster1 = pd.DataFrame()
    cluster2 = pd.DataFrame()
    combined_cluster = pd.DataFrame()

    # Add stocks to each cluster.
    for i in range(len(header_list)):
        if(cluster_result[i]) == num1:
            cluster1[header_list[i]] = temp_set[header_list[i]]
            combined_cluster[header_list[i]] = temp_set[header_list[i]]
        if(cluster_result[i]) == num2:
            cluster2[header_list[i]] = temp_set[header_list[i]]
            combined_cluster[header_list[i]] = temp_set[header_list[i]]

    cluster1_centroid = cluster1.mean(axis=1)
    cluster2_centroid = cluster2.mean(axis=1)
    combined_centroid = combined_cluster.mean(axis=1)


    # new error must greater than old
    old_error = 0
    new_error = 0

    attr_num = len(combined_centroid)
    combined_headers = list(combined_cluster)
    cluster1_headers = list(cluster1)
    cluster2_headers = list(cluster2)


    for i in range(len(combined_headers)):
        for j in range(attr_num):
            new_error += math.pow(combined_centroid[j] - combined_cluster[combined_headers[i]][j],2)

    for i in range(len(cluster1_headers)):
        for j in range(attr_num):
            old_error += math.pow(cluster1_centroid[j] - cluster1[cluster1_headers[i]][j],2)

    for i in range(len(cluster2_headers)):
        for j in range(attr_num):
            old_error += math.pow(cluster2_centroid[j] - cluster2[cluster2_headers[i]][j],2)

    return new_error - old_error










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


data_set = read_data('FTSE100_Data_Norm1.csv')
target_num = 10

sample_data = data_set.sample(25,random_state = 0)


[cluster_result,cluster_centroids,num_cluster] = hierarchical_based_Kmeans(data_set,5,target_num,15)


# clusters_visualization(sample_data,cluster_result,target_num)
for i in range (target_num):
    single_cluster_visualization(data_set,cluster_result,i)


print(silhouette_score_evaluation(data_set,cluster_result))






