# k-means-data-clustering
The unsupervised learning using K-Means clustering to analyze mall customer segmentation. Utilizing Python, Scikit-learn, and Matplotlib, the project includes steps to load and visualize the dataset, fit K-Means, determine optimal number of clusters using Elbow Method, visualize clusters, evaluate clustering performance with the Silhouette Score. 

 Code for Clustering with K-Means:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create sample data (since Mall_Customers.csv might not be available)
# You can replace this with: data = pd.read_csv('Mall_Customers.csv')
np.random.seed(42)
n_customers = 200

data = pd.DataFrame({
    'CustomerID': range(1, n_customers + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_customers),
    'Age': np.random.randint(18, 70, n_customers),
    'Annual Income (k$)': np.random.randint(15, 137, n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
})

# Step 2: Load and explore dataset
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Initial data visualization
plt.figure(figsize=(15, 5))

# Subplot 1: Annual Income vs Spending Score
plt.subplot(1, 3, 1)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           alpha=0.7, c='blue', s=60)
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, alpha=0.3)

# Subplot 2: Age distribution
plt.subplot(1, 3, 2)
plt.hist(data['Age'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Subplot 3: Gender distribution
plt.subplot(1, 3, 3)
gender_counts = data['Gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
        colors=['lightblue', 'lightpink'])
plt.title('Gender Distribution')

plt.tight_layout()
plt.show()

# Step 4: Feature selection and preprocessing
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures selected for clustering:")
print(X.head())
print("\nFeature statistics after scaling:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Standard deviation: {X_scaled.std(axis=0)}")

# Step 5: Elbow Method to find optimal K
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Plot Elbow Method
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='red')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.tight_layout()
plt.show()

# Print silhouette scores
for k, score in zip(K_range, silhouette_scores):
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Step 6: Apply K-Means clustering with optimal K
optimal_k = 5

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++', n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

data['Cluster'] = cluster_labels

cluster_centers_scaled = kmeans_final.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

print(f"\nClustering completed with K={optimal_k}")
print(f"Cluster centers (original scale):")
print(pd.DataFrame(cluster_centers, columns=X.columns))

print(f"\nCluster distribution:")
print(data['Cluster'].value_counts().sort_index())

# Step 7: Visualize clustering results
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           alpha=0.7, c='gray', s=60)
plt.title('Original Data')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=60)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Spending Score (1-100)'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=60)
plt.title('Age vs Spending Score by Cluster')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 8: Evaluate clustering performance
final_silhouette_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nFinal Silhouette Score: {final_silhouette_score:.3f}")

if final_silhouette_score > 0.5:
    print("Excellent clustering structure")
elif final_silhouette_score > 0.25:
    print("Good clustering structure")
else:
    print("Weak clustering structure")

# Analyze each cluster
print("\nCluster Analysis:")
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    print(f"\nCluster {i} ({len(cluster_data)} customers):")
    print(f"  Average Age: {cluster_data['Age'].mean():.1f}")
    print(f"  Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"  Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    print(f"  Gender distribution: {cluster_data['Gender'].value_counts().to_dict()}")

# Step 9: Optional PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42)
pca_labels = kmeans_pca.fit_predict(X_pca)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=60)
plt.title('Original Feature Space')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for i in range(optimal_k):
    mask = pca_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=60)
plt.title('PCA Feature Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.3f}")

   Explanation of this Code:
   
        => Step 1: Create Sample Data
                       - The code generates a synthetic dataset with 200 customers, including attributes such as CustomerID, Gender, Age, Annual Income, and Spending Score.
                       - The np.random.seed(42) ensures that the random numbers generated are reproducible.

        => Step 2: Load and Explore Dataset
                        - The shape, first few rows, info, statistical summary, and missing values of the dataset are printed to understand its structure and contents.

        => Step 3: Initial Data Visualization
                         Three visualizations are created:
                                    - A scatter plot of Annual Income vs. Spending Score.
                                    - A histogram showing the distribution of Age.
                                    - A pie chart displaying the distribution of Gender.

        => Step 4: Feature Selection and Preprocessing
                        - The features selected for clustering are Annual Income and Spending Score.
                                   These features are standardized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1, which is important for K-Means clustering.
                                   
         => Step 5: Elbow Method to Find Optimal K
                        - The Elbow Method is used to determine the optimal number of clusters (K) by calculating the Within-Cluster Sum of Squares (WCSS) and silhouette scores for K values ranging from 2 to 10. 
                        - Two plots are generated: one for WCSS and another for silhouette scores against the number of clusters.

         => Step 6: Apply K-Means Clustering with Optimal K
                        - The optimal number of clusters is set to 5 (based on previous analysis).
                        - K-Means clustering is applied, and the cluster labels are assigned to the original dataset.
                        - The cluster centers are calculated and transformed back to the original scale.

         => Step 7: Visualize Clustering Results
                          Three visualizations are created:
                                       - A scatter plot of the original data.
                                       - A scatter plot of the clustered data with centroids marked.
                                       - A scatter plot of Age vs. Spending Score colored by cluster.

         => Step 8: Evaluate Clustering Performance
                            - The final silhouette score is calculated to evaluate the clustering quality.
                             - Based on the silhouette score, a message is printed indicating the clustering structure's quality (excellent, good, or weak).

         => Step 9: Optional PCA for Dimensionality Reduction
                  - Principal Component Analysis (PCA) is performed to reduce the dimensionality of the data to two components.
                  - K-Means clustering is applied again on the PCA-transformed data.
            Three visualizations are created:
                        - Clusters in the original feature space.
                          - Clusters in the PCA feature space.
                            - A bar plot showing the explained variance ratio of the principal components.

         => Summary of Results:
                   The explained variance ratio from PCA is printed, indicating how much variance is captured by each principal component, along with the total variance explained.

            => This code provides a comprehensive approach to customer segmentation using K-Means clustering, including data exploration, visualization, and evaluation of clustering performance. It also demonstrates the use of PCA for dimensionality reduction, which can be useful for visualizing high-dimensional data.

  About the DataSet:
            - The dataset used in this task is a synthetic representation of customer data typically found in retail or mall environments. Here’s a brief overview of its structure and attrib
      
  Dataset Overview:
            -Number of Samples: 200 customers
  Attributes:
        1. CustomerID: A unique identifier for each customer (ranging from 1 to 200).
        2. Gender: The gender of the customer, which can be either 'Male' or 'Female'. This attribute is randomly assigned.
        3. Age: The age of the customer, generated randomly within the range of 18 to 70 years.
        4. Annual Income: The annual income of the customer, represented in thousands of dollars, generated randomly between 15 and 137.
        5. Spending Score (1-100): A score assigned to the customer based on their spending behavior, ranging from 1 to 100, generated randomly.

 => Purpose of the Dataset:
         The dataset is designed to simulate real-world customer data for the purpose of performing customer segmentation using K-Means clustering. By analyzing attributes such as age, income, and spending score, businesses can identify distinct customer groups, which can help in tailoring marketing strategies, improving customer service, and enhancing product offerings.

 => Characteristics:
             - Randomly Generated: The dataset is synthetic, meaning it does not represent real customers but is generated to mimic the characteristics of a typical customer base.
             - Diversity: The dataset includes a mix of genders, ages, income levels, and spending scores, allowing for a variety of clustering outcomes.
             - No Missing Values: Since the data is generated, there are no missing values, simplifying the analysis process.

This dataset serves as a practical example for demonstrating clustering techniques and data analysis methods in a controlled environment.


