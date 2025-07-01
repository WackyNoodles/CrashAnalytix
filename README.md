# CrashAnalytix Accident Detection System

This project is an Accident Detection System with multiple modules to handle various aspects of accident analysis, including accident detection, severity detection, object detection within accidents, and license plate detection.

## Overview

The project is divided into four main modules:

1. **Accident Detection**: Detects if an accident has occurred based on the input images or videos.
2. **Severity Detection**: Assesses the severity of the accident (minor, major, or critical).
3. **Object in Accident Detection**: Identifies objects (vehicles, pedestrians, etc.) present at the scene of the accident.
4. **License Plate Detection**: Detects and extracts license plate information from vehicles involved in the accident.

## Prerequisites

Before running the project, make sure you have the following installed:

- **Frontend**: Node.js and Yarn
- **Backend**: Python 3.12.8 and MongoDB

### Backend Requirements

1. **Python Libraries**: You will need to install the necessary Python dependencies using `pip`. The required dependencies are listed in the `requirements.txt` file in the backend directory.

2. **Models**: You will need to download pre-trained models to enable the detection features. These models can be downloaded from the following Google Drive link:

   [Download Models](https://drive.google.com/drive/folders/1Nia4YTmaevQj0hxsTHPuBN-uObfTyTGB?usp=sharing)

   Once downloaded, place them in the `backend` folder of the project.

## Setup Instructions

### 1. Clone the Repository
- git clone https://github.com/WackyNoodles/CrashAnalytix.git

### 2. Frontend Setup
- cd frontend
- yarn install

### 3. Backend Setup
- cd backend
- pip install -r requirements.txt

- Download the models from the provided Google Drive link and place them in the backend folder.

## Deployment
### 1. Deploy Backend
- cd backend
- python app.py

### 2. Deploy Frontend
- cd frontend
- yarn run dev
- This will start the development server for the frontend. You can access the application in your browser at http://localhost:3000 (or whatever the specified port is).

## Testing
- There are some testing videos in lpr_vids and test_vids folder.