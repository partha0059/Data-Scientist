# 🌸 Iris Flower Segmentation using Hierarchical Clustering

This project is a Streamlit web application that demonstrates **Hierarchical Clustering** (Agglomerative method) on the Iris dataset. It provides an intuitive user interface to input flower measurements and predicts the cluster assignment based on pre-trained models.

## 🌟 Features
- **Interactive UI**: Clean, light-themed interface built with Streamlit.
- **Real-time Prediction**: Instantly predicts the cluster of an Iris flower based on its Sepal and Petal dimensions using distance to cluster centroids.
- **Hierarchical Clustering**: Utilizes Agglomerative Clustering and Euclidean distance to assign new data points to clusters.

## 🚀 Live Demo
You can run this application locally or deploy it to Streamlit Community Cloud.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/partha0059/Hierarchial-Clustering.git
   cd Hierarchial-Clustering
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
- `app.py`: Main Streamlit application file containing the UI and prediction logic.
- `hierarchical_iris_model.pkl`: Pre-trained Agglomerative Clustering model.
- `iris_scaler.pkl`: StandardScaler fitted on the Iris dataset.
- `cluster_centroids.npy`: Saved cluster centroids used for prediction.
- `.streamlit/config.toml`: Configuration file for the light theme styling.
- `requirements.txt`: List of Python dependencies required to run the project.

## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a pull request.