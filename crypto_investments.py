#!/usr/bin/env python
# coding: utf-8

# # Module 10 Application
# 
# ## Challenge: Crypto Clustering
# 
# In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.
# 
# The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Import the Data (provided in the starter code)
# * Prepare the Data (provided in the starter code)
# * Find the Best Value for `k` Using the Original Data
# * Cluster Cryptocurrencies with K-means Using the Original Data
# * Optimize Clusters with Principal Component Analysis
# * Find the Best Value for `k` Using the PCA Data
# * Cluster the Cryptocurrencies with K-means Using the PCA Data
# * Visualize and Compare the Results

# ### Import the Data
# 
# This section imports the data into a new DataFrame. It follows these steps:
# 
# 1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.
# 
# 2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.
# 
# 
# > **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 

# In[27]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kmeans_elbow import kmeans_elbow


# In[28]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[29]:


# Generate summary statistics
df_market_data.describe()


# In[30]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data
# 
# This section prepares the data before running the K-Means algorithm. It follows these steps:
# 
# 1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.
# 
# 2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
# 

# In[31]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)


# In[32]:


# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()


# ---

# ### Find the Best Value for k Using the Original Data
# 
# In this section, you will use the elbow method to find the best value for `k`.
# 
# 1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following question: What is the best value for `k`?

# In[33]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
scaled_plot = kmeans_elbow(df=df_market_data_scaled)

scaled_plot


# #### Answer the following question: What is the best value for k?
# **Question:** What is the best value for `k`?
# 
# **Answer:** # 4

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data
# 
# In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the original data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.
# 
# 4. Create a copy of the original data and add a new column with the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[34]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[35]:


# Fit the K-Means model using the scaled data
fitted_model = model.fit(df_market_data_scaled)


# In[36]:


# Predict the clusters to group the cryptocurrencies using the scaled data
prediction = fitted_model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
prediction


# In[37]:


# Create a copy of the DataFrame
df_market_data_scaled_copy = df_market_data_scaled.copy()


# In[38]:


# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_copy['clusters'] = prediction

# Display sample data
df_market_data_scaled_copy


# In[39]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
clusters_plot = df_market_data_scaled_copy.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="clusters",
    hover_cols="coin_id"
)

clusters_plot


# ---

# ### Optimize Clusters with Principal Component Analysis
# 
# In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.
# 
# 1. Create a PCA model instance and set `n_components=3`.
# 
# 2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 
# 
# 3. Retrieve the explained variance to determine how much information can be attributed to each principal component.
# 
# 4. Answer the following question: What is the total explained variance of the three principal components?
# 
# 5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

# In[40]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[41]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
df_market_data_scaled_copy_pca = pca.fit_transform(df_market_data_scaled_copy)

# View the first five rows of the DataFrame. 
df_market_data_scaled_copy_pca[:5]


# In[42]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
explained_variance_ratio = pca.explained_variance_ratio_

display(explained_variance_ratio)

total_explained_variance = explained_variance_ratio.sum()

display(total_explained_variance)


# #### Answer the following question: What is the total explained variance of the three principal components?
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** # 0.8899113677327128

# In[43]:


# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
df_market_data_scaled_pca = pd.DataFrame(
    df_market_data_scaled_copy_pca,
    columns=['PCA1', 'PCA2', 'PCA3']
)

# Copy the crypto names from the original data
df_market_data_scaled_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled_pca = df_market_data_scaled_pca.set_index("coin_id")

# Display sample data
df_market_data_scaled_pca.head()


# ---

# ### Find the Best Value for k Using the PCA Data
# 
# In this section, you will use the elbow method to find the best value for `k` using the PCA data.
# 
# 1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

# In[44]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
pca_plot = kmeans_elbow(df=df_market_data_scaled_pca)

pca_plot


# #### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** # 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** # No

# ---

# ### Cluster Cryptocurrencies with K-means Using the PCA Data
# 
# In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the PCA data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.
# 
# 4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[45]:


# Initialize the K-Means model using the best value for k
pca_kmeans_model = KMeans(n_clusters=4)


# In[46]:


# Fit the K-Means model using the PCA data
fitted_pca_kmeans_model = pca_kmeans_model.fit(df_market_data_scaled_pca)


# In[47]:


# Predict the clusters to group the cryptocurrencies using the PCA data
predicted_pca_kmeans_model = fitted_pca_kmeans_model.predict(df_market_data_scaled_pca)

# View the resulting array of cluster values.
predicted_pca_kmeans_model


# In[48]:


# Create a copy of the DataFrame with the PCA data
df_market_data_scaled_pca_copy = df_market_data_scaled_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_pca_copy['clusters'] = predicted_pca_kmeans_model

# Display sample data
df_market_data_scaled_pca_copy


# In[49]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
pca_clusters_plot = df_market_data_scaled_pca_copy.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="clusters",
    hover_cols="coin_id"
)

pca_clusters_plot


# ---

# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.
# 
# 1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.
# 
# 2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.
# 
# 3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# > **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).

# In[50]:


# Composite plot to contrast the Elbow curves
composite_elbow_plot = scaled_plot + pca_plot

composite_elbow_plot


# In[51]:


# Compoosite plot to contrast the clusters
composite_scatter_plot = clusters_plot + pca_clusters_plot

composite_scatter_plot


# #### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** # Using fewer features can make it easier to identify which dimensions, or combination thereof, clearly categorize the data. I.e. for the scaled data scatter plot it's not obvious that the chosen clusters make sense because only 2 out of many features are used for analysis. A PCA analysis could help qualify the chosen clusters from the scaled analysis.
