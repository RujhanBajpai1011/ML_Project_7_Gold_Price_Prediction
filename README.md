# 💰 Gold Price Prediction

This project focuses on predicting the price of gold (GLD) using historical data of other related assets such as SPX (S&P 500), USO (United States Oil Fund), SLV (iShares Silver Trust), and EUR/USD exchange rates. A Random Forest Regressor model is employed for this prediction task.

## **📊 Dataset**

The dataset used is gld_price_data.csv. It contains daily historical data for:

* **Date**: The date of the observation.
* **SPX**: Price of S&P 500 index.
* **GLD**: Price of GLD (Gold ETF - our target variable).
* **USO**: Price of USO (United States Oil Fund ETF).
* **SLV**: Price of SLV (iShares Silver Trust ETF).
* **EUR/USD**: Euro to US Dollar exchange rate.

## **✨ Features**

* **Data Loading and Initial Inspection**: Loads the dataset into a pandas DataFrame and provides initial insights into its structure using head(), tail(), shape, and info().

* **Missing Value Check**: Confirms that there are no missing values in the dataset.

* **Statistical Measures**: Generates descriptive statistics of the data, including mean, standard deviation, min, max, and quartiles for each numerical feature.

* **Correlation Analysis**:
  * Calculates the correlation matrix between all numerical features.
  * Visualizes the correlations using a heatmap from seaborn to understand relationships between GLD and other variables.
  * Specifically checks and prints the correlation values of GLD with other features.

* **Data Distribution Visualization**: Displays the distribution of the GLD (Gold Price) using a distribution plot (distplot from seaborn).

* **Feature and Target Separation**: Splits the data into features (X) and the target variable (Y), where GLD is the target. The 'Date' column is dropped as it's not a direct feature for the regression model.

* **Data Splitting**: Divides the dataset into training and testing sets (80% training, 20% testing) to evaluate model performance.

* **Model Training**: Trains a Random Forest Regressor model on the preprocessed training data.

* **Model Evaluation**: Evaluates the model's performance using the R-squared error on both the training and test datasets.

* **Actual vs. Predicted Price Visualization**: Compares the actual GLD prices with the predicted prices from the model using scatter plots for both training and test data.

## **🛠️ Technologies Used**

* **Python**

* **pandas**: For data loading and manipulation.

* **numpy**: For numerical operations.

* **matplotlib.pyplot**: For creating static, interactive, and animated visualizations.

* **seaborn**: For statistical data visualization and creating the heatmap and distribution plot.

* **scikit-learn**: For machine learning tasks, including:
  * train_test_split: For splitting data.
  * RandomForestRegressor: For the regression model.
  * metrics: For evaluating model performance (e.g., R-squared error).

## **📦 Requirements**

To run this project, you will need the following Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## **🚀 Getting Started**

To get a copy of this project up and running on your local machine, follow these steps.

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

1. **Place the dataset**: Ensure the gld_price_data.csv file is located in the same directory as the Jupyter notebook (Gold Price Prediction.ipynb).

2. **Run the Jupyter Notebook**: Open and execute the cells in the Gold Price Prediction.ipynb notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, Google Colab).

The notebook will:

* Load and preprocess the gold price data.
* Perform exploratory data analysis, including correlation and distribution plots.
* Train the Random Forest Regressor model.
* Output the R-squared error scores for training and test datasets.
* Display scatter plots comparing actual vs. predicted prices.

## **📈 Results**

The notebook outputs the R-squared error scores for the Random Forest Regressor model on both the training and test datasets. These scores indicate how well the model captures the variance in gold prices.

* **R-squared Value (Training Data)**: Approximately 0.999 (very high, indicating a strong fit on training data)
* **R-squared Value (Test Data)**: Approximately 0.989 (high, indicating good generalization to unseen data)

The high R-squared values suggest that the Random Forest Regressor is a robust model for predicting gold prices based on the provided features.

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
