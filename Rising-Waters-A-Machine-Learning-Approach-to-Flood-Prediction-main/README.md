# Rising-Waters-A-Machine-Learning-Approach-to-Flood-Prediction

# 🌊 Rising Waters: A Machine Learning Approach to Flood Prediction

Floods are among the most devastating natural disasters, causing severe damage to lives, property, and the environment. Early prediction of floods plays a crucial role in disaster preparedness and risk mitigation. This project presents a **machine learning–based flood prediction system** that analyzes weather and rainfall data to predict the likelihood of flood occurrence.


## 📌 Project Overview

The **Rising Waters** project uses historical weather and rainfall data to train a machine learning model capable of predicting flood events. The system is deployed as a **web-based application** where users can input real-time weather parameters and receive flood risk predictions instantly. The goal of this project is to provide an **early warning mechanism** that is simple, reliable, and easy to use.


## ⚙️ Technologies Used

* **Programming Language:** Python
* **Machine Learning Algorithm:** Random Forest Classifier
* **Backend Framework:** Flask
* **Frontend:** HTML, CSS
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
* **Development Environment:** VS Code


## 📊 Dataset Description

The dataset consists of **115 records** with **11 attributes**, including weather parameters such as temperature, humidity, cloud cover, seasonal rainfall, and annual rainfall. Each record is labeled as **Flood or No Flood**, which serves as the target variable for training the model. The dataset was collected from publicly available online sources and preprocessed before model training.


## 🧠 Machine Learning Model

* The **Random Forest algorithm** is used for flood prediction due to its robustness and high accuracy.
* The dataset is split into training and testing sets.
* Class balancing is applied to improve prediction performance for flood cases.
* The trained model is saved and reused for real-time predictions.


## 🌐 System Architecture

1. User enters weather details through the web interface
2. Frontend sends input data to the Flask backend
3. Backend loads the trained machine learning model
4. Model predicts flood probability
5. Result is displayed on the web dashboard


## 🖥️ Web Application Features

* Simple and clean user interface
* Input fields for temperature, humidity, and cloud cover
* Real-time flood prediction output
* Clear indication of **Flood Expected** or **No Flood**


## ✅ Key Advantages

* Early flood prediction and warning
* Easy-to-use web interface
* Efficient handling of real-world data
* Scalable and extendable system
* Helps in disaster preparedness and decision-making


## 🚀 How to Run the Project

1. Clone the repository
2. Install required libraries using `pip install -r requirements.txt`
3. Run the Flask application using

   python app/app.py

4. Open the browser and visit


   http://127.0.0.1:5000


## 📌 Conclusion

This project demonstrates how machine learning can be effectively used to predict floods using weather data. By integrating a trained model with a web-based application, the system provides an accessible and practical solution for flood risk assessment and early warning.


Just tell me 👍
