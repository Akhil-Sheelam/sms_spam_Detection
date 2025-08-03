# sms_spam_Detection
SMS Spam Detection is a machine learning project that classifies SMS messages as spam or ham (non-spam) using a neural network built with TensorFlow and Keras. The model is trained on the SMS Spam Collection Dataset, which contains over 5,000 labeled SMS messages. 

📩 SMS Spam Detection using TensorFlow
This project detects whether an SMS message is spam or ham (not spam) using a deep learning model built with TensorFlow and Keras.<br>

🚀 Features
Preprocessing and tokenizing text data<br>
Building a neural network model using TensorFlow<br>
Training and validating the model<br>
Saving/loading model and tokenizer<br>
Predicting new messages (via main.py)<br>
Visualizing accuracy and loss over training epochs<br>

🧠 Model Architecture<br>
Embedding Layer: 5000 vocab size, 16 dimensions<br>
GlobalAveragePooling1D<br>
Dense Layer: 24 units, ReLU<br>
Output Layer: 1 unit, Sigmoid<br>

🗂️ Project Structure<br>
sms-spam-detection/<br>
│<br>
├── spam.csv # Dataset (SMS spam collection)<br>
├── train_model.py # Script to preprocess data and train model<br>
├── main.py # Script to load model & tokenizer, predict SMS text<br>
├── spam_model.h5 # Saved trained model<br>
├── tokenizer.pkl # Saved tokenizer for text preprocessing<br>
├── requirements.txt # All required dependencies<br>
└── README.md # Project documentation<br>

🧪 How to Use<br>
##1. Clone the repo<br>

git clone https://github.com/your-username/sms-spam-detection.git<br>
cd sms-spam-detection<br>

##2.Create and activate a virtual environment<br>
python -m venv venv<br>
venv\Scripts\activate # On Windows<br>

##3. Predict new SMS messages<br>
python main.py<br>

📦 Requirements<br>
Python 3.10<br>

TensorFlow 2.14.0<br>
NumPy 1.26.4<br>
scikit-learn<br>
pandas<br>
matplotlib<br>

✍️ Author<br>
Akhil sheelam

