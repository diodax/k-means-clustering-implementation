import numpy as np
import random
import math
from scipy import spatial
from statistics import mean


def dot_product(a, b):
    """
    Returns the dot product between two vector arrays
    """
    d_sum = 0.0
    for ((idx,), val) in np.ndenumerate(a):
        d_sum += float(val) * float(b[idx])
    return d_sum


def l2_norm(a):
    return math.sqrt(dot_product(a, a))


def cosine_similarity(a, b):
    """
        Returns the cosine similarity between two vector arrays
    """
    numerator = dot_product(a, b)
    denominator = (l2_norm(a) * l2_norm(b))
    return np.nan_to_num(np.divide(numerator, denominator))


def get_cosine_distance(a, b):
    # return 1 - cosine_similarity(a, b)
    return spatial.distance.cosine(a, b)


class KMeans:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}
        self.classes = {}

    def fit(self, data):
        # initialize the centroids, a random sample 'k' elements in the dataset will be our initial centroids
        data_sample = random.sample(list(data), self.k)
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data_sample[i]

        print("Initialize fitting with " + str(self.k) + " centroids")

        # begin iterations
        for i in range(self.max_iterations):
            print("Beginning iteration " + str(i) + " of the K-Means algorithm...")
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            # find the cosine distance between the point and each centroid and pick the closest one
            for features in data:
                classification = self.pred(features)
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            # average the cluster data points to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            is_optimal = True

            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    is_optimal = False

            # If the results change their positions less than our tolerance value, break out of the loop
            if is_optimal:
                print("Optimal centroids have been found after " + str(i) + " iterations, stopping...")
                break

    def pred(self, data):
        """Predicts the assigned cluster for the given data point"""
        distances = [get_cosine_distance(data, self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def get_clusters(self, vector_dict):
        """
        Given a dictionary with the IDs of the vectors, returns the IDs classified in the clusters.
        Used to generate the results report, given that it requires writing the user story Id instead
        of the features vector.
        """
        clusters = {}
        for i in range(self.k):
            clusters[i] = []
            # Add the vectors from self.classes to clusters, with the help of vector_dict
            for vector in self.classes[i]:
                vector_id = vector_dict[tuple(vector)]
                clusters[i].append(vector_id)
        return clusters

    def get_sse_score(self):
        """
        Returns the Sum of Squared Error (SSE) of the clusters.
        """
        sse_score = 0
        # Iterate through each cluster
        for i in range(self.k):
            # Inside each cluster, get the distance of each vector with the centroid
            for vector in self.classes[i]:
                # Calculate distance here
                distance = get_cosine_distance(vector, self.centroids[i])
                pow_distance = distance**2
                sse_score += pow_distance
        return sse_score

    def get_msc_avg(self):
        """
        Returns the Mean Silhouette Coefficient (MSC) of the clusters.
        """
        coefficients = []
        # Iterate through each cluster
        for i in range(self.k):
            # Inside each cluster, get the distance of each vector with the centroid
            for vector in self.classes[i]:
                a_values = []
                # Iterate through the points inside the current cluster
                for point in self.classes[i]:
                    distance = get_cosine_distance(vector, point)
                    a_values.append(distance)
                avg_a = mean(a_values)

                # Iterate through the points in the other clusters
                remaining_clusters = [key for key, value in self.classes.items() if key != i]
                avg_b_list = []
                for key in remaining_clusters:
                    b_values = []
                    for point in self.classes[key]:
                        distance = get_cosine_distance(vector, point)
                        b_values.append(distance)
                    avg_b = mean(b_values)
                    avg_b_list.append(avg_b)
                min_b = min(avg_b_list)

                # Calculate the silhouette coefficient for that point
                s_point = (min_b - avg_a) / float(max(avg_a, min_b))
                coefficients.append(s_point)

        # Get the average of all the coefficients
        msc_score_avg = mean(coefficients)
        return msc_score_avg
