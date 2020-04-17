import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabaz_score
import statistics
from sklearn.manifold import TSNE


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

    fig_title = 'K-means Clustering - ' + str(cluster_num) + ' clusters, S&P500, Normalisation 1, 2018'
    file_name = 'K-means Clustering - ' + str(cluster_num) + ' clusters, S&P500, Normalisation 1, 2018'
    fig = df_mean.plot(grid = True, title = fig_title)
    fig.set_xlabel('Date')
    fig.set_ylabel('Percentage')
    # fig.set_xticklabels(['','January',''])
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

    fig_title = 'K-means Clustering - Cluster' + str(cluster_num)
    file_name = 'K-means Clustering - Cluster' + str(cluster_num)
    fig = temp_df.plot(grid = True, title = fig_title)
    fig.set_xlabel('Date')
    fig.set_ylabel('Percentage')
    fig = fig.get_figure()
    file_path = '../Graph/'
    # fig.savefig(file_path + file_name)
    plt.show()


def silhouette_score_evaluation(data_set,cluster_result):
    return silhouette_score(data_set, cluster_result, metric= 'euclidean')

def calinski_harabaz_score_evaluation(data_set,cluster_result):
    return calinski_harabaz_score(data_set,cluster_result)

def davies_bouldin_score_evaluation(data_set,cluster_result):
    return davies_bouldin_score(data_set, cluster_result)



data_set = read_data('../Dataset/FTSE100_Data_Norm1.csv')
num = 26
cluster_model = KMeans(n_clusters=num).fit(data_set)
for i in range (num):
    single_cluster_visualization(data_set,cluster_model.labels_,i)

# for i in range(10):
#     cluster_model = KMeans(n_clusters=10).fit(data_set)
#     print(silhouette_score_evaluation(data_set,cluster_model.labels_))
# for i in range(81,100):
#     print(i)
#     result = 0
#     result_list = []
#     for j in range(20):
#         cluster_model = KMeans(n_clusters=i).fit(data_set)
#         result = silhouette_score_evaluation(data_set,cluster_model.labels_)
#         result_list.append(result)
#     print(statistics.stdev(result_list))


#clusters_visualization(data_set,cluster_model.labels_,num_cluster)
# for i in range(2,26):
#     print(i)
#     cluster_model = KMeans(n_clusters=i).fit(data_set)
#     result = silhouette_score_evaluation(data_set,cluster_model.labels_)
#     print(result)

#
# for i in range (num_cluster):
#     single_cluster_visualization(data_set,cluster_model.labels_,i)



# Method used to decreate the dimensions.
# data_TSNE = TSNE(learning_rate=100).fit_transform(data_set)
#
# plt.figure(figsize=(12,8))
# for i in range(2,6):
#     k = KMeans(n_clusters=i,max_iter=1000).fit_predict(data_set)
#     colors = ([['red','blue','black','yellow','green'][i] for i in k])
#     plt.subplot(219+i)
#     plt.scatter(data_TSNE[:,0],data_TSNE[:,1],c=colors,s=10)
#     plt.title('K-medoids Resul of '.format(str(i)))
#
# plt.show()



# clusters_visualization(data_set,cluster_model.labels_,num_cluster)
# cluster_model = KMeans(n_clusters=5).fit(data_set)

# for i in range(2,26):
#     result = 0
#     for j in range(50):
#         cluster_model = KMeans(n_clusters=i).fit(data_set)
#         result += silhouette_score_evaluation(data_set,cluster_model.labels_)
#     average = result / 50
#     print(average)





