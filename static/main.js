document.addEventListener('DOMContentLoaded', function () {
    // Global variables
    let dataPoints = [];
    let centroids = [];
    let labels = [];
    let initMethod = 'random';
    let manualCentroids = [];
    let nClusters = 4;
    let isKMeansInitialized = false; 

    // Get references to DOM elements
    const initMethodSelect = document.getElementById('init-method');
    const generateDataBtn = document.getElementById('generate-data');
    const stepBtn = document.getElementById('step');
    const runBtn = document.getElementById('run');
    const resetBtn = document.getElementById('reset');
    const nClustersInput = document.getElementById('n-clusters');
    const plotDiv = document.getElementById('plot');

    // Event listeners
    initMethodSelect.addEventListener('change', function () {
        initMethod = this.value;
        manualCentroids = []; // Clear manual centroids when switching methods
        centroids = []; // Clear centroids to reset the state
        labels = []; // Clear labels to reset the state
        isKMeansInitialized = false; // Reset the flag
        plotData(); // Re-plot to reset the graph
    });

    nClustersInput.addEventListener('change', function () {
        nClusters = parseInt(this.value);
        manualCentroids = []; // Reset centroids
        centroids = []; // Reset centroids
        labels = []; // Reset labels
        isKMeansInitialized = false; // Reset the flag
        plotData(); // Re-plot to reflect changes
    });

    generateDataBtn.addEventListener('click', generateData);
    stepBtn.addEventListener('click', stepAlgorithm);
    runBtn.addEventListener('click', runAlgorithm);
    resetBtn.addEventListener('click', resetAlgorithm);

    // Function to generate a new dataset
    function generateData() {
        fetch('/generate_data', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            dataPoints = data.data_points;
            centroids = [];
            labels = [];
            manualCentroids = [];
            isKMeansInitialized = false;
            plotData();
        })
        .catch(error => {
            console.error('Error generating data:', error);
        });
    }

    function initializeKMeansIfNeeded() {
        // Check if KMeans has already been initialized
        if (isKMeansInitialized) {
            // Already initialized, no need to reinitialize
            return Promise.resolve(true);
        }

        // Debugging logs to check manual centroids
        console.log('Initializing KMeans with:', {
            init_method: initMethod,
            n_clusters: nClusters,
            initial_centroids: initMethod === 'manual' ? manualCentroids : null
        });

        // Otherwise, initialize KMeans
        return fetch('/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                'init_method': initMethod,
                'n_clusters': nClusters,
                'initial_centroids': initMethod === 'manual' ? manualCentroids : null
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                centroids = data.centroids;
                labels = [];
                isKMeansInitialized = true;
                plotData();
                console.log('Initialization successful.');
                return true;  // Initialization successful
            } else {
                console.error('Initialization error:', data.message);
                return false;  // Initialization failed
            }
        })
        .catch(error => {
            console.error('Error during initialization:', error);
            return false; // Initialization failed
        });
    }

    // Function to perform one step of KMeans
    function stepAlgorithm() {
        if (initMethod === 'manual' && manualCentroids.length < nClusters) {
            alert('Please select all centroids before proceeding.');
            return;
        }

        initializeKMeansIfNeeded().then(initialized => {
            if (!initialized) return;  // Do not proceed if initialization failed

            fetch('/step', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    centroids = data.centroids;
                    labels = data.labels;
                    plotData();

                    // Check if KMeans has converged
                    if (data.converged) {
                        alert('KMeans has converged.');
                    }
                } else {
                    console.error('Step error:', data.message);
                }
            })
            .catch(error => {
                console.error('Error during stepping:', error);
            });
        });
    }

    // Function to run KMeans to convergence
    function runAlgorithm() {
        if (initMethod === 'manual' && manualCentroids.length < nClusters) {
            alert('Please select all centroids before proceeding.');
            return;
        }

        initializeKMeansIfNeeded().then(initialized => {
            if (!initialized) return;  // Do not proceed if initialization failed

            fetch('/run', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    centroids = data.centroids;
                    labels = data.labels;
                    plotData();
                } else {
                    console.error('Run error:', data.message);
                }
            })
            .catch(error => {
                console.error('Error during running:', error);
            });
        });
    }

    // Function to reset the algorithm
    function resetAlgorithm() {
        fetch('/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                centroids = [];
                labels = [];
                manualCentroids = [];
                isKMeansInitialized = false; 
                plotData();  // Re-plot to reflect the reset state
            } else {
                console.error('Reset error:', data.message);
            }
        })
        .catch(error => {
            console.error('Error during reset:', error);
        });
    }

    // Function to plot data and centroids
    function plotData() {
        let data = [];

        // Plot data points
        if (dataPoints && dataPoints.length > 0) {
            let x = dataPoints.map(p => p[0]);
            let y = dataPoints.map(p => p[1]);
            let marker = { size: 6 };

            if (labels && labels.length > 0) {
                marker.color = labels;
                marker.colorscale = 'Viridis';
            } else {
                marker.color = 'blue';
            }

            data.push({
                x: x,
                y: y,
                mode: 'markers',
                type: 'scatter',
                marker: marker,
                name: 'Data Points'
            });
        }

        // Plot centroids
        if (centroids && centroids.length > 0) {
            let x = centroids.map(p => p[0]);
            let y = centroids.map(p => p[1]);

            data.push({
                x: x,
                y: y,
                mode: 'markers',
                type: 'scatter',
                marker: { color: 'red', size: 12, symbol: 'x' },
                name: 'Centroids'
            });
        }

        let layout = {
            title: 'KMeans Clustering',
            clickmode: 'event',
            xaxis: { range: [-10, 10] },
            yaxis: { range: [-10, 10] }
        };

        Plotly.newPlot('plot', data, layout);
    }

    // Add click event to the plot 
    plotDiv.addEventListener('click', function(event) {
        if (initMethod === 'manual' && manualCentroids.length < nClusters) {
            let xaxis = plotDiv._fullLayout.xaxis;
            let yaxis = plotDiv._fullLayout.yaxis;

            let bbox = plotDiv.getBoundingClientRect();
            let xPixels = event.clientX - bbox.left;
            let yPixels = event.clientY - bbox.top;

            // Adjust for margins
            let l = plotDiv._fullLayout.margin.l;
            let t = plotDiv._fullLayout.margin.t;

            xPixels = xPixels - l;
            yPixels = yPixels - t;

            let xData = xaxis.range[0] + (xPixels / xaxis._length) * (xaxis.range[1] - xaxis.range[0]);
            let yData = yaxis.range[0] + (1 - yPixels / yaxis._length) * (yaxis.range[1] - yaxis.range[0]);

            console.log('Clicked at x=' + xData + ', y=' + yData);

            // Add selected point as centroid
            manualCentroids.push([xData, yData]);
            centroids = manualCentroids;
            plotData(); // Update plot to show new centroid

            if (manualCentroids.length === nClusters) {
                initializeKMeansIfNeeded();
            }
        } else if (manualCentroids.length >= nClusters) {
            console.log('Maximum number of centroids selected.');
        }
    });

    // Initial plot
    plotData();
});
