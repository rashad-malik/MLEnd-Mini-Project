# **Who's Fibbing? - Detecting Deception in Spoken Narratives**

This repository contains the code and resources for a mini-project that explores detecting deception in spoken narratives using machine learning models. The project processes audio data, extracts meaningful features, and builds predictive models to classify whether a story is true or deceptive.

---

## **Project Overview**

This project uses the **MLEnd Deception Dataset**, a collection of audio files, to identify vocal patterns indicative of deception. The key stages of the project include:

1. **Data Preprocessing**: Converting audio files into consistent formats, trimming, and segmenting into 30-second chunks.
2. **Feature Extraction**: Deriving acoustic features such as pauses, pitch range, and variability to serve as inputs for machine learning models.
3. **Model Training and Validation**: Employing models such as Logistic Regression, Support Vector Machines (SVM), and k-Nearest Neighbours (k-NN) for classification, with hyperparameter optimisation and cross-validation.
4. **Evaluation**: Testing the best-performing model on unseen data to measure accuracy and interpret results.

---

## **Repository Structure**

```
MLEnd Mini Project/
├── notebooks/
│   ├── Experimental code.ipynb           # Miscellaneous exploration and test code
│   ├── Whos_fibbing.ipynb                # Main project notebook
│   ├── MLEnd Mini Project - Starter Kit Code.ipynb # Original starter code (for reference)
├── variables/
│   ├── X_train.npy                       # Extracted features for training set
│   ├── y_train.npy                       # Labels for training set
│   ├── groups_train.npy                  # Groups for training set (to prevent data leakage)
│   ├── X_test.npy                        # Extracted features for test set
│   ├── y_test.npy                        # Labels for test set
│   ├── groups_test.npy                   # Groups for test set
├── Whos_fibbing.html                     # Main project notebook in HTML
├── README.md                             # Project documentation
```

The main notebook, **"Whos fibbing.ipynb"**, contains the entire workflow, from data preparation to model evaluation. You can view this in as a HTML file.

---

## **Setup Instructions**

### **Dependencies**
Install the required Python libraries before running the project:
```bash
pip install numpy pandas scikit-learn tqdm librosa matplotlib
```

### **Downloading the Dataset**
This project requires additional data that is not included in the repository due to size constraints. To obtain the dataset, install the **MLEnd** library (version `1.0.0.4`):

```bash
pip install mlend==1.0.0.4
```

Then, use the following commands to download and load the dataset:
```python
import mlend
from mlend import download_deception_small, deception_small_load

# Download the dataset
datadir = download_deception_small(save_to='MLEnd', subset={}, verbose=1, overwrite=False)
```

---

## **How to Run**

1. **Data Setup**: Ensure you’ve downloaded the data as described above.
2. **Run the Notebook**:
   - Open the main notebook file `Whos fibbing.ipynb`.
   - Follow the cells sequentially to preprocess the data, extract features, and train the models.
3. **Evaluate the Model**:
   - The notebook includes code to train the model on the training set and evaluate it on the test set.
   - Outputs include accuracy, confusion matrices, and feature importance visualisations.

---

## **Key Features**
- **Feature Extraction**: Includes silence detection, pitch analysis, and temporal feature engineering.
- **Model Pipelines**: Built-in support for Logistic Regression, SVM, and k-NN models with normalisation and hyperparameter tuning.
- **Cross-Validation**: Uses GroupKFold to prevent data leakage.
- **Comprehensive Evaluation**: Provides accuracy metrics, classification reports, and confusion matrices.

---

## **Future Work**
- Experiment with ensemble methods like Random Forest or Gradient Boosting.
- Incorporate additional features such as emotional tone or linguistic cues.
- Improve silence detection thresholds for diverse audio environments.
- Address class imbalance through oversampling or weighted loss functions.

---

## **Acknowledgements**

This project has been completed as part of the **Principles of Machine Learning** module, which is part of the **Data Science and Artificial Intelligence** course at **Queen Mary University of London**. The project has provided a practical opportunity to apply theoretical concepts from the module to a real-world machine learning problem.

Additionally, special thanks to the creators of the **MLEnd** library, Dr. Jesús Requena Carrión and Dr. Nikesh Bajaj, for facilitating access to the **MLEnd Deception Dataset**, which served as the foundation for this work.