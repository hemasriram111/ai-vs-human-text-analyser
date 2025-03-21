# AI vs Human Text Classifier

This project is a machine learning model that classifies whether a given piece of text is written by a human or generated by an AI. The model is trained using a Logistic Regression algorithm with TF-IDF vectorization for text feature extraction. The project also includes a Streamlit-based web application for real-time predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Frontend Application](#frontend-application)
- [Contributing](#contributing)
- [License](#license)

## Model accuracy

Accuracy: 0.9940993565733168

Classification Report:
               precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     61112
         1.0       0.99      0.99      0.99     36335

    accuracy                           0.99     97447
   macro avg       0.99      0.99      0.99     97447
weighted avg       0.99      0.99      0.99     97447

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hemasriram111/ai-vs-human-text-analyser.git
   cd ai-vs-human-text-analyser
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model and vectorizer:**
   - Ensure that the `ai_vs_human_model.pkl` and `tfidf_vectorizer.pkl` files are in the project directory.

## Usage

### Running the Streamlit App

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser. You can enter a piece of text in the provided text area, and the application will predict whether the text is AI-generated or human-written.

### Example

1. Enter a piece of text in the text area.
2. Click the "Predict" button.
3. The application will display the prediction result.

## Model Training

The model is trained using a dataset containing both human-written and AI-generated text. The training process involves the following steps:

1. **Data Loading:** The dataset is loaded from a CSV file.
2. **Text Vectorization:** The text data is converted into TF-IDF features.
3. **Model Training:** A Logistic Regression model is trained with class weights to handle imbalanced data.
4. **Evaluation:** The model's performance is evaluated using accuracy and a classification report.

### Training Script

The training script (`model.py`) can be run to train the model from scratch:

```bash
python model.py
```

This script will:
- Load the dataset.
- Preprocess the text data.
- Train the model.
- Save the trained model and vectorizer to disk.

## Frontend Application

The frontend application is built using Streamlit, a popular framework for creating data science web applications. The application allows users to input text and get real-time predictions from the trained model.

### Features

- **Text Input:** Users can enter any piece of text.
- **Prediction:** The application predicts whether the text is AI-generated or human-written.
- **Result Display:** The prediction result is displayed with a success message.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
