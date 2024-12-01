# DeepSentiment
This repository provides a stock prediction app which predicts the stock prices by taking historical price trends and social media sentiments.

# Project Structure
/src/main.py = Code for gradio app.
/src/utils = This directory contains various code including data exploration and model training, etc
deployment = Contain the Dockerfile
monitoring = Constains the prometehus file.

# Features
**Stock Prediction**: Predict future stock prices using an LSTM model trained on historical stock data and social media sentiment
**Model Training**: A Jupyter Notebook (`LSTM.ipynb`) that walks through the steps of training the LSTM model.
**Real-time Monitoring**: Metrics for monitoring the Gradio app's performance and stock prediction errors are tracked using Prometheus.
**Visualization**: Metrics are visualized in real-time using Prometheus and Grafana.
**Dockerized Deployment**: The app and monitoring stack are deployed using Docker and Docker Compose for easy setup and deployment.

# Prerequisites
Docker
Docker Compose
Python 3.7+
Jupyter Notebook (for training the model)

# Setup and Installation
1) Clone the repository
2) Navigate to deployment folder
3) Build the docker image using docker-compose build command
4) Start the container using docker-compose up command
5) The gradio app will be accessible at http://localhost:7860
6) Prometheus will be accessible at http://localhost:9090
7) Grafana will be available at http://localhost:3000
8) Stop the container using docker-compose down command

# Monitoring with Prometheus and Grafana
The app tracks various metrics using Prometheus (running on port 9090) and visualizes them in Grafana (running on port 3000). The metrics include:
1) Request count
2) Request latency
3) Model prediction error (RMSE)
4) Memory usage
5) Active requests
6) Prediction values distribution
7) Feedback ratings

# Prometheus Configuration
The prometheus.yml file in the monitoring directory defines the scrape configuration for Prometheus to gather metrics from the Gradio app

# Grafana Setup
Once Grafana is up and running, you need to configure it to use Prometheus as a data source:
1) Go to http://localhost:3000 (default login: admin / admin).
2) Add Prometheus as a data source (URL: http://prometheus:9090).
3) Import a pre-configured dashboard or create your own to visualize the app's metric

# Feedback
The app includes an interface for providing feedback. Your feedback is tracked using Prometheus counters and stored in a JSON file.



