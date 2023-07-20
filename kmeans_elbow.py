# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans


def kmeans_elbow(k: "list[int]" = list(range(1, 11)), df: "pd.DataFrame" = None):
    # Create an empty list to store the inertia values
    inertia = []

    # Create a for loop to compute the inertia with each possible value of k
    for i in k:
        model = KMeans(n_clusters=i, random_state=0)
        model.fit(df)
        inertia.append(model.inertia_)
        
    # Create a dictionary with the data to plot the elbow curve
    elbow_data = {
        "k": k,
        "inertia": inertia
    }

    # Create a DataFrame with the data to plot the elbow curve
    df_elbow = pd.DataFrame(elbow_data)
    
    # Plot the elbow curve
    plot = df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
    
    return plot

