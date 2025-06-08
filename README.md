# Song Genre Clustering Analysis

This project performs clustering analysis on Spotify song data to identify patterns and relationships between different music genres based on audio features.

## Project Overview

The analysis uses K-means clustering to group songs based on their audio features, with the following key components:

1. **Data Preprocessing**
   - Handles missing values and duplicates
   - Standardizes numerical features
   - Performs PCA for dimensionality reduction (2 components)

2. **Cluster Analysis**
   - Tests cluster numbers from 2 to 10 (max_k)
   - Determines optimal number of clusters using:
     - Elbow Method (15% threshold)
     - Silhouette Score Method (10% threshold)
   - Visualizes clustering metrics in real-time

3. **Visualization**
   - Correlation matrix of audio features
   - Interactive cluster analysis plots
   - Genre distribution across clusters
   - Feature importance analysis
   - Cluster feature distributions for key metrics:
     - danceability
     - energy
     - valence
     - acousticness

## Requirements

- Python 3.x
- Required packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd song-genre-clustering
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your `spotify_songs.csv` file in the project directory
2. Run the analysis:
   ```bash
   python script.py
   ```

## Data

The analysis uses the Spotify Songs dataset which includes various audio features such as:
- danceability
- energy
- valence
- acousticness
- instrumentalness
- liveness
- loudness
- speechiness
- tempo
- key
- mode
- duration_ms

## Output

The script generates several visualizations:
1. Correlation matrix of audio features
2. Cluster analysis plots:
   - Silhouette scores
   - Elbow method
3. Audio clusters visualization with centroids
4. Genre distribution heatmap
5. Feature importance for principal components
6. Box plots of key features by cluster

## Analysis Methods

### Optimal Cluster Selection
- **Elbow Method**: 
  - Uses 15% threshold of maximum change
  - Finds first point where rate of decrease stabilizes
- **Silhouette Score**: 
  - Uses 10% threshold of maximum change
  - Finds first point where score stabilizes

### Clustering
- Uses K-means clustering with PCA-reduced features (2 components)
- Random state set to 42 for reproducibility
- Visualizes clusters in 2D space using principal components
- Shows cluster centroids with red X markers

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
