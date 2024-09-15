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
