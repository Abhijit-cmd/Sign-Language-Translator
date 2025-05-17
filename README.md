    Sign Language Translator
Sign Language Translator is a deep learning-powered application that translates American Sign Language (ASL) gestures into text using a trained neural network model. Built with Python, TensorFlow/Keras, and OpenCV, this tool aims to improve communication for the hearing and speech impaired by recognizing hand gestures in real-time.

    Table of Contents
•	Overview
•	Features
•	Technologies Used
•	Project Structure
•	Installation
•	Usage
•	Model Details
•	Demo
•	Future Work
•	Contributing
•	License
•	Contact

    Overview
The Sign Language Translator processes real-time webcam input to recognize 26 ASL alphabet gestures and converts them into corresponding English letters, enabling users to communicate without speech.


    Features
•	Real-time webcam gesture recognition using OpenCV
•	Deep learning model trained on ASL alphabets
•	Translates gestures into English text
•	Lightweight and extensible for additional sign language support
•	Simple and user-friendly interface


    Technologies Used
•	Python
•	TensorFlow / Keras
•	OpenCV
•	Flask and versal for hosting and deployment
•	NumPy and Pandas

    Project Structure
sign_language_translator/
├── gesture_alphabet_model.h5          # Pretrained CNN model file
├── main.py                           # Main script to run the application
├── requirements.txt                  # Project dependencies
├── README.md                        # Project documentation
├── dataset/                         # Dataset (if applicable)
├── utils/                           # Utility functions and scripts
└── .gitignore                       # Git ignore rules

    Installation
1. Clone the repository:

```bash
git clone https://github.com/Abhijit-cmd/Sign-Language-Translator.git
cd Sign-Language-Translator
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Make sure to have your webcam ready for input.

Usage
Run the main script to start translating signs:

```bash
python main.py
```
The app will open your webcam, and start detecting ASL alphabets, displaying the corresponding text on screen in real-time.
 
 
     Model Details
- Model Type: Convolutional Neural Network (CNN)
- Trained on: ASL alphabet and gesture dataset
- Model file: `gesture_alphabet_model.h5` (stored with Git LFS due to size)

      Demo
 ![Screenshot 2025-05-14 173531](https://github.com/user-attachments/assets/6704fae0-776e-41aa-94fa-e94c027a33e6)
![Screenshot 2025-03-27 105228](https://github.com/user-attachments/assets/4cc247f7-6870-48b9-bf45-002043b6ac17)
https://github.com/user-attachments/assets/d8817e8e-fd6f-4f2b-bee1-da52ae7fb3d2

     Future Work
•	Expand to include common phrases and sentences
•	Improve accuracy with more training data
•	Add multi-language sign recognition
•	Create a mobile app version

    Contributing
Contributions are welcome! Please fork the repo and submit pull requests.
AS THIS WAS A GROUP PROJECT 
MADE BY 
ABHIJIT
DIPESH 
MEGHA
AMAN 
FT PRESIDENCY UNIVERSITY 

    License
This project is for academic purposes. Licensing terms can be added if open-sourced.

    Contact
Abhijit Deb
GitHub: https://github.com/Abhijit-cmd
Email: abhijitdeb063@gmail.com




