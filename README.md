# Brain Tumor Classification
## Overview
This is an end-to-end deep learning project focusing on developing a tool to predict the presence of brain tumors from MRI scans, adhering to MLOps best practices. It involves data collection and preprocessing with TensorFlow datasets, model training with TensorFlow, model serving and version management using TensorFlow Serving in a Docker image, and deployment through a FastAPI endpoint with a ReactJS front-end for user interaction. The end-to-end pipeling is shown in the architecture diagram below.

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

## Methodology

The project involved several steps from data collection to preprocessing, model building, and serving.

### 1. Data Collection

The dataset used  is available on **Kaggle**: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri). It contains MRI scans of brain tumors, divided into **training** and **testing** sets, each with four classes:

1. **No Tumor**
2. **Pituitary Tumor**
3. **Meningioma Tumor**
4. **Glioma Tumor**

# Brain Tumor Detection Model



### 2. Data Preprocessing
- **Image Conversion:** Images were transformed into tensors and converted to RGB format to ensure uniformity.
- **Resizing:** Each image was resized to 155x155 pixels, and a batch size of 32 was used during training.
- **Data Augmentation:** To improve generalization, random rotations and contrast adjustments were applied, generating artificial samples to handle different image orientations and contrasts.
- **Normalization:** All pixel values were scaled by dividing by 255, ensuring values fell between 0 and 1.

### 3. Model Building
A convolutional neural network with the following architecture was trained on the training data:
  - Four convolutional layers to capture complex patterns in MRI images.
  - One fully connected layer for classification, with a dropout rate of 0.5 to reduce overfitting.
    
The model currently achieves a 61% accuracy on the test dataset obtained from Kaggle, which was held out during model training for unbiased evaluation.

### 3. Model Versioning and Serving
Multiple versions of the model were saved throughout experimentation, and TensorFlow Serving was used to manage and serve them. The system was configured to automatically mount the latest model version within a TensorFlow Serving Docker container, ensuring seamless updates without manual intervention.

### 4. FastAPI EndPoint and React.js Frontend
A FastAPI application was developed to interact with TensorFlow Serving, allowing users to upload MRI images for inference and receive predictions from the trained model. The API supports batch processing, enabling efficient handling of multiple user requests simultaneously. This helps in scenarios where many users interact with the model at the same time. A user-friendly web interface built with React.js was integrated with the FastAPI backend. It enables users to easily upload MRI scans via their web browsers. Once the images are uploaded, the interface communicates with the FastAPI API, and the model predictions are displayed alongside the confidence scores. 


### Next Steps

**Model Improvements**
- **Exploring Advanced Architectures:** To improve the current model accuracy (61%), established architectures such as **ResNet** could be explored. These architectures have achieved good performance in similar tasks by allowing deeper models to extract more complex features without suffering from degradation in performance. Implementing such architectures could significantly improve feature extraction and classification accuracy.
- **Hyperparameter Tuning:** Experimenting with hyperparameters like learning rates, batch sizes, and dropout rates may enhance both model performance and generalization. 
- **Transfer Learning:** Leveraging pre-trained models and fine-tuning them on the specific dataset may not only accelerate training but also improve accuracy by utilizing the learned features from larger datasets.


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

