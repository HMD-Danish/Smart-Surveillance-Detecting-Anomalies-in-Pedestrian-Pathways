Smart Surveillance: Detecting Anomalies in Pedestrian Pathways
📌 Overview
This project implements a smart surveillance system that detects anomalies in pedestrian pathways using deep learning. The system combines Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and transfer learning to identify irregularities in pedestrian behavior and anomalies in static vehicle images. It is built with a Django backend and a responsive web-based frontend.

🧠 Core Features
Real-time surveillance and anomaly detection

Automatic feature extraction using CNN

Behavioral analysis using LSTM

Prediction using multiple ML models (Random Forest, Gradient Boosting, SVM)

Static image analysis for vehicle anomalies using CNN + Autoencoders

Web application for users and admins

⚙️ System Architecture
Frontend: HTML, CSS, JavaScript (Bootstrap)

Backend: Django Framework (Python)

Database: SQLite

ML Models: CNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Autoencoders

Deployment Environment:

Python 3.8+

Libraries: scikit-learn, TensorFlow/PyTorch, OpenCV, NumPy, Pandas

🛠️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone <repository-url>
cd smart-surveillance
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Django Server
bash
Copy
Edit
python manage.py runserver
🧪 Testing and Evaluation
Unit Testing, Integration Testing, and System Testing included

Performance metrics evaluated:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

Real-world robustness tested on noisy and low-resolution images

📂 Project Structure
bash
Copy
Edit
├── users/
│   └── views.py        
├── admins/
│   └── views.py      
├── templates/
│   └── HTML templates for UI
├── static/
│   └── CSS/JS files
├── fuelconsumption/
│   └── urls.py         
├── utility/
│   └── process_ml.py  
└── manage.py
📈 Future Enhancements
Integrate real-time video feed processing

Expand to smart city monitoring

Improve model training with larger datasets

Deploy to cloud platforms (AWS/GCP)

👩‍💻 Authors
Kummara Mahesh

Mandem Amrutha

Rajaka Pranay Krishna

H. Mohammad Danish

Avula Yashwanth Sai

Supervised by: Dr. B. M. G. Prasad, Professor & Dean, Department of CSE, ALITS
