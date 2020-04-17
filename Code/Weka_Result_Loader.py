# In this file, it contains methods used to load the result and then analyse the
# result produced by Weka.
import csv
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabaz_score
import pandas as pd

def data_loader(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        df = pd.DataFrame(list(reader))

        # The first row became headers
        new_header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data less the header row
        df.columns = new_header  # set the header row as the df header

        df = df.drop(columns='Instance_number')
        df = df.set_index('index')

        cluster_result = df['Cluster']
        df = df.drop(columns='Cluster')

        data_set = df.copy()

        return [data_set,cluster_result]

def silhouette_score_evaluation(data_set,cluster_result):
    return silhouette_score(data_set, cluster_result, metric= 'euclidean')

def calinski_harabaz_score_evaluation(data_set,cluster_result):
    return calinski_harabaz_score(data_set,cluster_result)

def davies_bouldin_score_evaluation(data_set,cluster_result):
    return davies_bouldin_score(data_set, cluster_result)


for i in range(4,26):
    print(i)
    file ='../EM result/' + str(i) + '.arff.csv'
    [data_set, cluster_result] = data_loader(file)
    print(calinski_harabaz_score_evaluation(data_set,cluster_result))