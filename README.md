# Epoch-Spring-Camp-Unsupervised-Learning
This is a GitHub Repository for the Epoch Spring Camp Unsupervised Learning Task of implementing K-means Clustering from scratch

**Problem Statement:**

**Objective**  

You have been provided with a dataset containing information about various pincodes across India, including their corresponding longitudes and latitudes (clustering_data.csv). Your task is to focus specifically on the pincodes of your home state. Here’s what you need to do:  

**1. Data Filtering:**  

Extract the entries corresponding to your home state from the dataset. Ensure you accurately filter out only those pincodes that belong to your home state.  

**2. Data Visualization:**  

You can utilize the longitude and latitude data to plot the geographical locations of these pincodes on a map (get creative!). This will help in visualizing the distribution of the pincodes across your state.  

**3. Clustering Analysis:**  

Implement the k-means clustering algorithm from scratch (do not use any pre-built k-means function).
Apply this algorithm to the longitude and latitude data of your filtered pincodes to identify distinct clusters within your state.  

**4. Inference and Insights:**  

Draw meaningful inferences from your clustering results. Analyze the characteristics of the clusters you identified and provide insights about the geographical distribution and potential implications.  

**5. Visualization and Preprocessing:**  

Use appropriate visualization techniques (like scatter plots, maps, etc.) to illustrate the clusters and any other relevant observations.
Ensure your data is preprocessed correctly before applying the k-means algorithm (this may include handling missing values,checking for duplicates etc.).  

**New Features Added**  
1. For the scatter plot of the Latitude and Longitude using matplotlib , I have tried to make something cool about the plot by making it display all the details regarding that place using external library mplcursors . Here I have added a feature to display all the details of a place when we hover the mouse cursor onto that point on the scatter plot.
2. I have tried to optimize the value of k in the K-Means clustering algorithm . So I have found out the optimal k value by plotting the elbow curve by calculating within-cluster sum of squares (WCSS) value for different k values.

**Inferences and Insights:**  
1. Optimal number of clusters i.e value of K determined by the elbow point of the curve is 2 for the given csv file.
2. Number of points clustered into 2 clustered are 373 and 8184 respectively.
3. Actually Andhra Pradesh is located between 12 to 19 degree North latitudes and 77 to 85 degree East longitude approximately. But there are some extreme outliers for some places ... There is one point at 800 North latitude which is impossible. Also some places have extremely low longitude. This suggests that there may be some mistakes in the dataset. So by using these K-Means Clustering , I think we may be able to find out these places as anomalies and this may also help us in verifying data once again.
4. By increasing the number of clusters , I think we may further subdivide the state into districts and this can be highly useful in real-world tasks like Business and Postal Departments , Disaster Management etc..

**Instructions to run the code on the machine:**  
1. Open a terminal (Command Prompt, Terminal, or Anaconda Prompt, depending on your OS and setup)
2. Navigate to the directory where your requirements.txt file is located:
   cd path/to/your/project
3. Run the following command: pip install -r requirements.txt (This installs all the required libraries in the machine).
4. After that , finally run the code using the command : python3 k_means_clustering.py
5. Note : All the files must be in the same directory.



