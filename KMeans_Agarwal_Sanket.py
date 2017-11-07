"""
@author:Sanket Agarwal
Python code for K-means clustering (k = 1-20) and reiterating 1000 times on each k-means
(for selecting random centroids initially 1000 times fo each k-means) for better accuracy
"""
import random as r
import math as m
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting graphs


class ClusterPrototype:  # Class for cluster prototype

    def __init__(self, centroid):
        """
        The constructor for ClusterPrototype
        :param centroid: Its the centroid of the cluster
        """
        self.data = []
        self.size = 0
        self.centroid = centroid

    def add_record(self, record):
        """
        This function is to add records to the cluster
        :param record: Eac individual record
        :return: None
        """
        self.data.append(record)
        self.size += 1

    def sse_calculation(self):
        """
        This method is to calculate sum of squared error for each cluster
        :return: None
        """
        sse = 0
        for row in self.data:
            sse += (row[0]-self.centroid[0])**2 + (row[1]-self.centroid[1])**2 + (row[2]-self.centroid[2])**2
        return sse

    def centroid_re_computation(self):
        """
        This method is for re calculation of centroid
        :return: None
        """
        self.centroid = []
        c1 = 0
        c2 = 0
        c3 = 0
        for row in self.data:
            c1 += row[0]
            c2 += row[1]
            c3 += row[2]
        if self.size != 0:
            self.centroid = [c1/self.size, c2/self.size, c3/self.size]

    def get_data(self):
        """
        Getter method for the data
        :return: Data in this cluster
        """
        return self.data

    def get_size(self):
        """
        Getter method for size of the cluster
        :return: Size of the cluster
        """
        return self.size

    def get_centroid(self):
        """
        Getter method for centroid
        :return: The centroid of the cluster
        """
        return self.centroid


class KCluster:  # This class is for each K- means

    def __init__(self):
        """
        Constructor for the KCluster
        """
        self.total_cluster_list = []
        self.identity = 0
        self.total_sse = 0

    def add_cluster(self, cluster):
        """
        This method is to add cluster to the present k-means
        :param cluster: Its the cluster to be added
        :return: None
        """
        self.total_cluster_list.append(cluster)
        self.identity += 1

    def centroid_recalculation(self):
        """
        This method re-calculates the centroid of each cluster in this k-means
        :return: None
        """
        for cluster in self.total_cluster_list:
            cluster.centroid_re_computation()
        centroid = [cluster.get_centroid() for cluster in self.total_cluster_list]
        return centroid

    def flush_KCluster(self):
        """
        This flushes the present k-means by removing all the cluster in it
        :return: None
        """
        self.total_cluster_list = []
        self.identity = 0

    def sse_total(self):
        """
        This method calculates the total sum of squared error (Intra and inter)
        :return: The total SSE
        """

        for cluster in self.total_cluster_list:
            self.total_sse += cluster.sse_calculation()
        return self.total_sse

    def get_cluster_list(self):
        """
        This is a getter method for the cluster list in this k-means
        :return:
        """
        return self.total_cluster_list

    def get_identity(self):
        """
        Getter method for this k-means (what is the value of k)
        :return: value of K
        """
        return self.identity


class HandlingClusterData:  # This class handles the data and does the actual clustering

    def __init__(self, file):
        """
        The constructor method for initializing cluster parameters
        :param file: The file
        """
        self.data_list = []  # entire data
        self.clusters = []  # List for clusters
        self.centroid = []  # Initial centroid. Its a list of centroids of the k cluster
        self.file = file
        with open(file) as f:
            count = 0
            for row in f:
                if count != 0:
                    self.data_list.append([float(_) for _ in row.split(',')])
                count += 1

    def initial_centroid(self, k):
        """
        Initializing the centroid randomly
        :param k: The value for k
        :return: Centroid
        """
        centroid = []

        for count in range(k):
            centroid.append(self.data_list[r.randint(0, len(self.data_list)-1)])
        return centroid

    def calculate_euclidean_dist(self, centroid, record):

        """
        Method to calculate the euclidean distance between two clusters
        :param com_1:  The list of the centroid of the cluster 1
        :param com_2:  The list of the centroid of the cluster 2
        :return: Euclidean distance
        """

        squared_diff_sum = 0.0
        for d in range(len(centroid)):
            squared_diff_sum += (centroid[d] - record[d])**2

        return m.sqrt(squared_diff_sum)

    def calculate_difference(self, c1, c2, threshold):
        """
        This method calculates the difference between the two centroids
        :param c1:
        :param c2:
        :param threshold:
        :return: Boolean true or false based on if the difference is greater than the threshold value
        """
        diff = []
        diff1 = 0
        diff2 = 0
        diff3 = 0
        size = len(c1)

        for i in range(len(c1)):
            if len(c1[i]) != 0 and len(c2[i]) != 0:
                diff1 += m.fabs(c1[i][0] - c2[i][0])
                diff2 += m.fabs(c1[i][1] - c2[i][1])
                diff3 += m.fabs(c1[i][2] - c2[i][2])
        diff = [diff1/size, diff2/size, diff3/size]
        if diff[0] < threshold and diff[1] < threshold and diff[2] < threshold:
            return True
        else:
            return False

    def k_means_clustering(self):
        """
        This is the method for doing the actual operations for clustering
        :return: None
        """

        for k in range(1, 21):
            min_sse = m.inf
            prev_cluster_type = None
            k_cluster_type = None
            print("k :"+str(k))

            for _ in range(1000):  # iterating 1000 times for each k-means (while selecting the random centroids) for better accuracy.
                centroid = self.initial_centroid(k)
                k_cluster_type = KCluster()

                while True:
                    for ind in range(k):  # Calculation for each k-means clustering
                        # print("k :"+str(k))
                        # print(ind)
                        # print(len(centroid))
                        c = ClusterPrototype(centroid[ind])

                        k_cluster_type.add_cluster(c)  # recording each k-means clustering

                    for row in self.data_list:  # dividing the data between the clusters based on euclidean distance
                        min_dist = m.inf
                        rec = None
                        for cluster in k_cluster_type.get_cluster_list():
                            dist = self.calculate_euclidean_dist(cluster.get_centroid(), row)
                            # cluster_curr = None
                            if dist < min_dist:
                                min_dist = dist
                                cluster_curr = cluster

                        if cluster_curr is not None:
                            cluster_curr.add_record(row)
                    prev_centroid = centroid
                    centroid = k_cluster_type.centroid_recalculation()
                    if len(centroid) != len(prev_centroid):
                        print('hello')
                    if self.calculate_difference(centroid, prev_centroid, threshold=0.1):
                        # self.clusters.append(k_cluster_type)
                        break
                    else:
                        k_cluster_type.flush_KCluster()
                sse_total = k_cluster_type.sse_total()
                if sse_total < min_sse:
                    min_sse = sse_total
                    prev_cluster_type = k_cluster_type
            if prev_cluster_type is not None:
                self.clusters.append(prev_cluster_type)
        print([cluster_type.sse_total() for cluster_type in self.clusters])

        '''for cluster_type in self.clusters:
            print("Cluster identity: "+str(cluster_type.get_identity()))
            count = 1
            for cluster in cluster_type.get_cluster_list():
                print("Cluster "+str(count))
                print("size: "+str(len(cluster.get_data())))
                count += 1'''

        self.plot_k_vs_sse(self.clusters)  # Plotting K v/s (sum of squared error)

    def plot_k_vs_sse(self, clusters):
        """
        This method is plotting K v/s sum of squared error(sse)
        :param clusters: Its the list of all the k-means
        :return: None
        """
        plt.figure().suptitle("K versus Sum of Squared Difference(SSE)", fontsize=20)
        plt.plot([k for k in range(1, 21)], [cluster_type.sse_total() for cluster_type in clusters], 'bo')
        plt.ylabel('Sum of Squared Error(SSE)')
        plt.xlabel('K')
        plt.grid(True)
        plt.show()


def main():
    """
    The main function for calling the HandlingData() method
    :return: None
    """

    file_name = input("Enter the file name:")

    h = HandlingClusterData(file_name)
    h.k_means_clustering()  # calling the method for dong the actual clustering

if __name__ == "__main__":
    main()


















