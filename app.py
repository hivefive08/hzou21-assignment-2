from flask import Flask, render_template, request, jsonify
import numpy as np
from assignment02 import KMeans

app = Flask(__name__)

# Global variables to hold the state
kmeans_instance = None
data_points = None

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/generate_data', methods=['POST'])
def generate_data():
    global data_points
    n_samples = 300
    data_points = np.random.uniform(low=-10, high=10, size=(n_samples, 2))
    data_points = data_points.tolist()  # Convert to list for JSON serialization

    return jsonify({'status': 'success', 'data_points': data_points})

@app.route('/initialize', methods=['POST'])
def initialize():
    global kmeans_instance, data_points
    try:
        init_method = request.json.get('init_method')
        n_clusters = request.json.get('n_clusters', 4)
        initial_centroids = request.json.get('initial_centroids', None)

        print(f"Received Initialization Request: Method={init_method}, n_clusters={n_clusters}, initial_centroids={initial_centroids}")

        if data_points is None:
            print("Error: Data not generated yet.")
            return jsonify({'status': 'error', 'message': 'Data not generated yet.'}), 400

        # Create the KMeans instance
        kmeans_instance = KMeans(n_clusters=n_clusters, init_method=init_method, max_iter=100)
        
        # Call the fit() method, which internally handles the initialization based on init_method
        if init_method == 'manual' and initial_centroids is not None:
            kmeans_instance.fit(np.array(data_points), initial_centroids=initial_centroids)
        else:
            kmeans_instance.fit(np.array(data_points))

        return jsonify({'status': 'success', 'centroids': kmeans_instance.centroids.tolist()})
    except Exception as e:
        print(f"Error during initialization: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/step', methods=['POST'])
def step():
    global kmeans_instance
    if kmeans_instance is None:
        return jsonify({'status': 'error', 'message': 'KMeans not initialized.'}), 400
    
    try:
        converged = kmeans_instance.step()  # Perform a single step
        centroids = kmeans_instance.centroids.tolist()
        labels = kmeans_instance.labels.tolist()
        return jsonify({'status': 'success', 'centroids': centroids, 'labels': labels, 'converged': converged})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/run', methods=['POST'])
def run():
    global kmeans_instance
    if kmeans_instance is None:
        return jsonify({'status': 'error', 'message': 'KMeans not initialized.'}), 400

    for _ in range(kmeans_instance.max_iter):
        old_centroids = kmeans_instance.centroids.copy()
        kmeans_instance.labels = kmeans_instance.assign_clusters()
        kmeans_instance.centroids = kmeans_instance.update_centroids()
        if np.allclose(kmeans_instance.centroids, old_centroids):
            break

    return jsonify({
        'status': 'success',
        'centroids': kmeans_instance.centroids.tolist(),
        'labels': kmeans_instance.labels.tolist(),
        'converged': True
    })

@app.route('/reset', methods=['POST'])
def reset():
    global kmeans_instance
    kmeans_instance = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
