# Sign Language Translator

**Sign Language Translator** is a deep learning-powered application that translates American Sign Language (ASL) gestures into text using a trained neural network model.  
Built with [Python](https://www.python.org/), [TensorFlow/Keras](https://www.tensorflow.org/), and [OpenCV](https://opencv.org/), this tool aims to improve communication for the hearing and speech impaired by recognizing hand gestures in real-time.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Demo](#-demo)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)


---

## 📝 Overview

The Sign Language Translator processes real-time webcam input to recognize **26 ASL alphabet gestures** and converts them into corresponding English letters, enabling users to communicate without speech.
---

## ✨ Features
- Real-time webcam gesture recognition using OpenCV  
- Deep learning model trained on ASL alphabets  
- Translates gestures into English text  
- Lightweight and extensible for additional sign language support  
- Simple and user-friendly interface  

---

---

## 🧪 Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask and Versal (for hosting and deployment)  
- NumPy and Pandas  

---
## 📁 Project Structure

```
sign_language_translator/
├── gesture_alphabet_model.h5     # Pretrained CNN model file
├── main.py                        # Main script to run the application
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
├── dataset/                       # Dataset (if applicable)
├── utils/                         # Utility functions and scripts
└── .gitignore                     # Git ignore rules
```

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Abhijit-cmd/Sign-Language-Translator.git
cd Sign-Language-Translator
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Ensure your webcam is ready for input.**

---
## ▶️ Usage

Run the following command to start the translator:

```bash
python main.py
```

The application will launch your webcam and start detecting ASL alphabets, displaying the corresponding text on-screen in real-time.

---
## 🧠 Model Details

- **Model Type:** Convolutional Neural Network (CNN)  
- **Trained On:** ASL alphabet and gesture dataset  
- **Model File:** `gesture_alphabet_model.h5` (stored with Git LFS due to size)  

---
## 🎥 Demo

### Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/6704fae0-776e-41aa-94fa-e94c027a33e6)  
![Screenshot 2](https://github.com/user-attachments/assets/4cc247f7-6870-48b9-bf45-002043b6ac17)  

### Video

🔗 [Demo Video Link](https://github.com/user-attachments/assets/d8817e8e-fd6f-4f2b-bee1-da52ae7fb3d2)

---
## 🚀 Future Work

- Expand to include common phrases and sentences  
- Improve accuracy with more training data  
- Add multi-language sign recognition  
- Develop a mobile application version  

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

### Group Members:
- Abhijit  
- Dipesh  
- Megha  
- Aman  

👨‍🎓 Project developed as part of an academic curriculum at **Presidency University**.

---

## 📜 License

This project is intended for academic purposes. Licensing terms can be added if open-sourced.

---

## 📬 Contact

**Abhijit Deb**  
- GitHub: [Abhijit-cmd](https://github.com/Abhijit-cmd)  
- Email: [abhijitdeb063@gmail.com](mailto:abhijitdeb063@gmail.com)  



