# wash_clustering_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df.dropna(subset=['Urban (%)', 'Rural (%)', 'National (%)'])

def normalize_features(df):
    features = df[['Urban (%)', 'Rural (%)', 'National (%)']].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled

def run_kmeans(scaled_data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(scaled_data)

def visualize_clusters(df, scaled_data, clusters):
    plt.figure(figsize=(10, 6))
    for cluster_id in sorted(df['Cluster'].unique()):
        points = scaled_data[clusters == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id}')

    for i, label in enumerate(df['Indicator']):
        plt.text(scaled_data[i, 0]+0.05, scaled_data[i, 1], label, fontsize=9)

    plt.xlabel("Urban (Standardized)")
    plt.ylabel("Rural (Standardized)")
    plt.title("K-Means Clustering of WASH Indicators (Kenya 2021)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    filepath = "WASH_Kenya_2021_Clustered.csv"
    df = load_data(filepath)
    scaled_data = normalize_features(df)
    clusters = run_kmeans(scaled_data)
    df['Cluster'] = clusters
    df.to_csv("WASH_Kenya_2021_Clustered_Output.csv", index=False)
    visualize_clusters(df, scaled_data, clusters)

if __name__ == "__main__":
    main()
