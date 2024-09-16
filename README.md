# Brain Tumor Classification
## Overview
This is an end-to-end deep learning project focusing on predicting brain tumors from MRI scans, adhering to MLOps best practices. It involves data collection and preprocessing with TensorFlow datasets, model training with TensorFlow, model serving and version management using TensorFlow Serving in a Docker image, and deployment through a FastAPI endpoint with a ReactJS front-end for user interaction. The end-to-end pipeling is shown in the architecture diagram below.

## Architecture Diagram
![End-to-End Architecture](Project%20Images/arch.png)
*Figure 1: End-to-End Architecture.*
## User Interface
Before discussing the problem and methodology, here’s a quick look at how the system works. The images below show the interface for uploading MRI scans and the model’s prediction results

### MRI Scan Upload Interface
![MRI Scan Upload Interface](Project%20Images/pg1.png)
*Figure 2: Interface for uploading MRI scans for tumor detection.*

### Tumor Detection Result
![Tumor Detection Result](Project%20Images/pg2.png)
*Figure 3: Prediction of Meningioma Tumor with confidence score.*

## Problem Statement
Interpreting MRI scans to detect brain tumors can be challenging because different brain diseases may appear similar. For example, an aggressive glioblastoma might resemble cancer that has spread to the brain from another part of the body. This is a problem because early and accurate diagnosis is crucial for ensuring proper treatment and improving patient outcomes. Therefore, a system that can accurately detect brain tumors from these scans can serve as a decision support tool, streamlining the diagnostic process and enabling faster, more informed treatment decisions.


## Setup Instructions

1. Clone the Repository
    ```bash
    git clone https://github.com/Abdul-AA/Brain-Tumor-Classifier.git
    cd Brain-Tumor-Classifier
    ```

2. Run TensorFlow Serving with Docker


    ```bash
    docker run -t --rm -p 8501:8501 \
      -v "/<your-root-path>/Brain-Tumor-Classifier:/Brain-Tumor-Classifier" \
      tensorflow/serving --rest_api_port=8501 --model_base_path=/Brain-Tumor-Classifier/saved_models
    ```

    Make sure the TensorFlow Serving instance is running on `http://localhost:8501`.

3. Install Backend Dependencies and Run FastAPI

    ```bash
    pip3 install -r api/requirements.txt
    cd api
    uvicorn main:app --reload --host localhost --port 8000
    ```

4. Set Up and Run the Frontend

    Navigate to the frontend directory, install dependencies, fix potential issues, and start the frontend server:

    ```bash
    cd ../frontend
    npm install --from-lock-json
    npm audit fix
    npm run start
    ```

    This will start the frontend on `http://localhost:3000`.

5. Upload MRI Scans for Prediction

    Once the application is running, you can **drag and drop MRI images** into the app to get a tumor classification prediction. The results, including the predicted tumor type and confidence level, will be displayed in the app.

---

### Important Notes:
- **Replace `<your-root-path>` with the absolute path** to the project directory on your machine wherever applicable.
- Make sure TensorFlow Serving is running on `http://localhost:8501` and the FastAPI backend is running on `http://localhost:8000` before interacting with the frontend.

