# Docker Deployment – Iris ML Classifier

## 1. Objective

To containerize the Iris Machine Learning application using Docker and deploy it as a lightweight HTTP prediction service.

## 2. Docker Architecture

The application uses a multi-stage Docker build:

* **Builder stage:** Installs Python dependencies and trains the Random Forest model.
* **Runtime stage:** Copies the trained model, label encoder, dependencies, and serving script into the final image.

This approach separates model training from application serving.

## 3. Technologies Used

* Docker and Docker Desktop
* Python 3.11
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Git and GitHub

## 4. Build Docker Image

Command:

```bash
docker build -t iris-classifier:v1 .
```

Result: Docker image built successfully.

Image name: `iris-classifier:v1`

## 5. Run Container

Command:

```bash
docker run -d --name iris-classifier-container -p 8000:8000 iris-classifier:v1
```

The application runs on port 8000.

## 6. API Endpoints

### Health Check

Endpoint: `GET /health`

Successful response:

```json
{"status": "healthy"}
```

### Prediction

Endpoint: `POST /predict`

The API accepts seven Iris features and returns the predicted species and class probabilities.

Successful test prediction:

```json
{
  "prediction": "setosa",
  "probabilities": [1.0, 0.0, 0.0]
}
```

Invalid input returns HTTP 400 with a list of missing features.

## 7. Container Verification

The following checks were completed successfully:

* Docker image built successfully.
* Container started successfully.
* Docker health status: `healthy`.
* Health endpoint returned HTTP 200.
* Valid prediction returned HTTP 200.
* Invalid input returned HTTP 400.
* Container logs confirmed successful application startup and API requests.

## 8. Image Details

Image: `iris-classifier:v1`

Reported image content size: approximately 156 MB.

The Dockerfile uses a multi-stage build to keep the runtime image separate from the model-training environment.

## 9. Conclusion

The Iris ML application was successfully containerized using Docker. The application provides health-check and prediction endpoints, and its container passed the required functional and health checks.
