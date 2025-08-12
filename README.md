# 🍷 Wine Quality Prediction

This project aims to predict the quality of red wine based on various physicochemical properties using a Random Forest Classifier. It involves data loading, exploratory data analysis, preprocessing, model training, and evaluation.

## **📊 Dataset**

The dataset used in this project is winequality-red.csv, which contains 11 input variables (physicochemical properties) and one output variable (quality) for various red wine samples.

Features of the dataset include:

* fixed acidity
* volatile acidity
* citric acid
* residual sugar
* chlorides
* free sulfur dioxide
* total sulfur dioxide
* density
* pH
* sulphates
* alcohol
* quality (the target variable, ranging from 3 to 8)

## **✨ Features**

* **Data Loading and Initial Inspection**: Loads the dataset into a Pandas DataFrame and provides a quick look at its dimensions (.shape) and the first few rows (.head()).

* **Missing Value Check**: Confirms the absence of missing values in the dataset.

* **Statistical Measures**: Generates descriptive statistics of the dataset, including mean, standard deviation, min, max, and quartiles for each feature.

* **Exploratory Data Analysis (EDA)**:
  * Visualizes the distribution of quality using a count plot to show the number of samples for each quality score.
  * Uses bar plots to show the relationship between quality and volatile acidity, as well as quality and citric acid.
  * Generates a heatmap to visualize the correlation matrix between all features in the dataset, helping to understand feature relationships.

* **Data Preprocessing**:
  * Converts the quality target variable into a binary classification problem: 'Good Quality Wine' (1) for quality scores 7 and 8, and 'Bad Quality Wine' (0) for quality scores 3, 4, 5, and 6.
  * Separates features (X) from the target variable (Y).

* **Data Splitting**: Divides the preprocessed data into training and testing sets (80% training, 20% testing).

* **Model Training**: Trains a Random Forest Classifier model on the training data.

* **Model Evaluation**: Calculates and prints the accuracy score of the trained model on both the training and test datasets.

* **Predictive System**: Includes a function or code snippet to make predictions on new, unseen data, classifying wine as "Good Quality Wine" or "Bad Quality Wine".

## **🛠️ Technologies Used**

* **Python**

* **pandas**: For data manipulation and analysis.

* **numpy**: For numerical operations.

* **matplotlib.pyplot**: For creating static visualizations.

* **seaborn**: For enhanced statistical data visualizations.

* **scikit-learn**: For machine learning tasks, specifically:
  * train_test_split: For splitting data.
  * RandomForestClassifier: For the classification model.
  * accuracy_score: For model evaluation.

## **📦 Requirements**

To run this project, you will need the following Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## **🚀 Getting Started**

To get a copy of this project up and running on your local machine, follow these simple steps.

### **Installation**

1. **Clone the repository (if applicable):**

```
git clone <repository_url>
cd <repository_name>
```

2. **Install the required Python packages:**

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Usage**

1. **Place the dataset**: Ensure the winequality-red.csv file is in the same directory as the Jupyter notebook (Wine_Quality_Prediction.ipynb).

2. **Run the Jupyter Notebook**: Open and execute the cells in the Wine_Quality_Prediction.ipynb notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, Google Colab).

The notebook will:

* Load and preprocess the wine quality data.
* Perform exploratory data analysis and visualize key relationships.
* Train the Random Forest Classifier.
* Output the accuracy scores.
* Demonstrate a predictive system for new data inputs.

## **📈 Results**

The notebook provides accuracy scores for the Random Forest Classifier. Based on the provided snippets, the model achieves high accuracy on both training and test data after classifying quality into two categories (good/bad). For instance:

* **Accuracy on training data**: Approximately 1.0 (100%)
* **Accuracy on test data**: Approximately 0.925 (92.5%)

These results suggest that the Random Forest model is highly effective in predicting wine quality based on the given features.

## **🧑‍💻 Contributing**

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes.
4. Commit your changes (git commit -m 'Add new feature').
5. Push to the branch (git push origin feature/your-feature-name).
6. Open a Pull Request.

## **📄 License**

This project is open-source and available under the MIT License.
