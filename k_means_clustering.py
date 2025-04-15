import numpy as np
import matplotlib.pyplot as plt
import mplcursors # For interactive cursor that helps to add hover and click interaction to MatPlotlib plots

import pandas as pd


# Step 1 and 5:  Data filtering and preprocessing
dataset = pd.read_csv("clustering_data.csv",low_memory=False) # Reads csv file intp a Pandas DataFrame


# print(dataset.columns) # Prints the column names of the DataFrame
home_state = "ANDHRA PRADESH"
home_state_data_filtered = dataset[dataset["StateName"] == home_state].copy() # Filters the DataFrame for the specified state
# print(home_state_data_filtered) # Prints the filtered DataFrame
# There is some error occuring during scatter plots which is causing TypeError so trying to rectify that
# Making sure that the Latitude and Longitude columns are of float type
home_state_data_filtered.loc[:,"Latitude"] = pd.to_numeric(home_state_data_filtered["Latitude"],errors="coerce")
home_state_data_filtered.loc[:,"Longitude"] = pd.to_numeric(home_state_data_filtered["Longitude"],errors="coerce")

# Drop invalid rows
home_state_data_filtered.dropna(subset=["Latitude", "Longitude"], inplace=True)

# Drop duplicates
home_state_data_filtered.drop_duplicates(subset=["Pincode","Latitude","Longitude"],inplace=True)


# This mplcursors library is very useful in adding interactive features to static matplotlib plots
# Here I have added a creative feature for data visualization to show the details of a place that I have
# plotted on scatter plot by showing up details for hovering the mouse cursor.
# Step 3 : Clustering Analysis

class KMeansClustering:
    def __init__(self,k=3,max_iterations = 100,max_centroid_movement = 0.001):
        self.k = k
        self.max_iterations = max_iterations
        self.max_centroid_movement = max_centroid_movement
    def fit(self,data):
        self.data = data
        # This data is converted into a NumPy array of domensions (num_samples , num_features) before fitting
        self.centroids = data[np.random.choice(data.shape[0],self.k,replace=False)]
        # The above step is for selecting k random points as centroids initially
        for _ in range(self.max_iterations):
            self.labels = []
            for datapoint in data:
                distances = []
                for centroid in self.centroids:
                    distance = np.linalg.norm(datapoint - centroid)
                    distances.append(distance)
                min_distance = np.argmin(np.array(distances))
                self.labels.append(min_distance)
            labels = np.array(self.labels)
            # The above step is calculating the distance of each data point from each centroid
            # and storing the distances in a list
            
        

            # The above step is assigning each data point to the nearest centroid
            new_centroids = []
            for i in range(self.k):
                points = []
                for j in range(len(self.data)):
                    if(self.labels[j] == i):
                        points.append(data[j])
                if points != []:

                    new_centroids.append(np.mean(points,axis=0)) # Axis=0 makes the np.mean for both latitudes and longitudes
                    # If axis=0 is not used , it is giving ValueError in broadcasting

                else :
                    new_centroids.append(self.centroids[i]) # Avoids NaN values if points list is empty
            
            new_centroids = np.array(new_centroids)

            # The above step is for calculating the new centroids by taking the mean of the points assigned to each centroid

            centroid_movement = np.linalg.norm(new_centroids - self.centroids)
            # The above step is for calculating the movement of the centroids
            if centroid_movement < self.max_centroid_movement:
                break
            else :
                self.centroids = new_centroids

    def predict(self,X):
        predictions = []
        for datapoint in X:
            distances = []
            for centroid in self.centroids:
                distance = np.linalg.norm(centroid - datapoint)
                distances.append(distance)
            predictions.append(np.argmin(np.array(distances)))
        return np.array(predictions)
    

# Step 4  and 5 : Inference and Insights , Data Visusalisation , Preprocessing and Plotting scatter plots
# Fitting the model by giving the csv file as input
# For this , we have to make a (num_samples , num_features) array
# For simplicity , num_features = 2( Latitude and Longitude)
home_state_data_filtered_for_clustering = home_state_data_filtered[["Longitude" , "Latitude"]].values
# print(home_state_data_filtered_for_clustering)
# Compute within-cluster sum of squares (WCSS) for different values of k
wcss = []
K_range = range(1, 11)

for k in K_range:
    model = KMeansClustering(k=k)
    model.fit(home_state_data_filtered_for_clustering)

    # Compute WCSS for this k
    total_wcss = 0
    for i in range(len(home_state_data_filtered_for_clustering)):
        centroid = model.centroids[model.predict([home_state_data_filtered_for_clustering[i]])[0]]
        distance = np.linalg.norm(home_state_data_filtered_for_clustering[i] - centroid)
        total_wcss += distance ** 2
    wcss.append(total_wcss)


def find_elbow_point(k_values, wcss):
    # Vector from first to last point
    line_vec = np.array([k_values[-1] - k_values[0], wcss[-1] - wcss[0]])
    # Normalize the vector
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    # Calculate the distance of each point to the line
    for i in range(len(k_values)):
        # Vector from first point to current point
        point = np.array([k_values[i] - k_values[0], wcss[i] - wcss[0]])
        # Project point onto line
        proj = np.dot(point, line_vec_norm) * line_vec_norm
        dist_vec = point - proj
        distance = np.linalg.norm(dist_vec)
        distances.append(distance)
    # Find the index of the maximum distance
    # This index corresponds to the elbow point
    # The elbow point is the point where the distance to the line is maximum
    # This is the optimal number of clusters
    return k_values[np.argmax(distances)]

elbow_k = find_elbow_point(list(K_range), wcss)
print(f"Optimal number of clusters (elbow point): {elbow_k}")



k_means_clustering_model = KMeansClustering(k=elbow_k,max_iterations=100,max_centroid_movement=0.001)
k_means_clustering_model.fit(home_state_data_filtered_for_clustering)
# print(k_means_clustering_model.centroids)


# Count the number of points in each cluster
predictions = k_means_clustering_model.predict(home_state_data_filtered_for_clustering)
count_zeroes = np.count_nonzero(predictions == 0)
count_ones = np.count_nonzero(predictions == 1)

print(f"Number of points in cluster 0: {count_zeroes}")
print(f"Number of points in cluster 1: {count_ones}")


# Step 2 : Data Visualization

# Use matplotlib scatterplot
fig,axs = plt.subplots(1,3,figsize=(15, 6))
axs1 = axs[0]
axs2 = axs[1]
axs3 = axs[2]
axs1.scatter(home_state_data_filtered['Longitude'], home_state_data_filtered['Latitude'], c='blue', marker='o')
axs1.set_title(f'Pincodes in {home_state}')
axs1.set_xlabel('Longitude')
axs1.set_ylabel('Latitude')
axs1.grid()

# Add an interactive cursor
cursor = mplcursors.cursor(hover=True)

@cursor.connect("add")
def on_add(sel):
    index = sel.index
    circlename = home_state_data_filtered.iloc[index]['CircleName']
    regionname = home_state_data_filtered.iloc[index]['RegionName']
    divisionname = home_state_data_filtered.iloc[index]['DivisionName']
    office_name = home_state_data_filtered.iloc[index]['OfficeName']
    office_type = home_state_data_filtered.iloc[index]['OfficeType']
    delivery = home_state_data_filtered.iloc[index]['Delivery']
    district = home_state_data_filtered.iloc[index]['District']
    state = home_state_data_filtered.iloc[index]['StateName']
    pincode = home_state_data_filtered.iloc[index]['Pincode']
    latitude = home_state_data_filtered.iloc[index]['Latitude']
    longitude = home_state_data_filtered.iloc[index]['Longitude']
    sel.annotation.set_text(f'Circle: {circlename}\nRegion: {regionname}\nDivision: {divisionname}\nOffice: {office_name}\nType: {office_type}\nDelivery: {delivery}\nDistrict: {district}\nState: {state}\nPincode: {pincode}\nLatitude: {latitude}\nLongitude: {longitude}')



# Plotting the clusters
axs2.scatter(home_state_data_filtered['Longitude'], home_state_data_filtered['Latitude'], c=predictions, cmap='viridis', marker='o')
axs2.scatter(k_means_clustering_model.centroids[:, 0], k_means_clustering_model.centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
axs2.set_title(f'Clusters in {home_state}')
axs2.set_xlabel('Longitude')
axs2.set_ylabel('Latitude')
axs2.legend()
axs2.grid()
# Add an interactive cursor
cursor2 = mplcursors.cursor(hover=True)
@cursor2.connect("add")
def on_add(sel):
    index = sel.index
    circlename = home_state_data_filtered.iloc[index]['CircleName']
    regionname = home_state_data_filtered.iloc[index]['RegionName']
    divisionname = home_state_data_filtered.iloc[index]['DivisionName']
    office_name = home_state_data_filtered.iloc[index]['OfficeName']
    office_type = home_state_data_filtered.iloc[index]['OfficeType']
    delivery = home_state_data_filtered.iloc[index]['Delivery']
    district = home_state_data_filtered.iloc[index]['District']
    state = home_state_data_filtered.iloc[index]['StateName']
    pincode = home_state_data_filtered.iloc[index]['Pincode']
    latitude = home_state_data_filtered.iloc[index]['Latitude']
    longitude = home_state_data_filtered.iloc[index]['Longitude']
    sel.annotation.set_text(f'Circle: {circlename}\nRegion: {regionname}\nDivision: {divisionname}\nOffice: {office_name}\nType: {office_type}\nDelivery: {delivery}\nDistrict: {district}\nState: {state}\nPincode: {pincode}\nLatitude: {latitude}\nLongitude: {longitude}')

# Extra feature : Plot and visualize an elbow curve in order to find the optimal number of clusters

axs3.plot(K_range, wcss, 'bo-')
axs3.set_xlabel('Number of clusters (k)')
axs3.set_ylabel('WCSS')
axs3.set_title('Elbow Curve for Optimal k value')
axs3.grid(True)
plt.show()

# Step 4 : Inferences -------->  Included in README.md file










    





