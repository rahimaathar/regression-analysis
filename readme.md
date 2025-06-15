# Regression Analysis Web App

A Flask-based web application for performing regression analysis on user-uploaded datasets. The app supports both Linear and Lasso regression, provides automated data preprocessing, visualizes regression results, and reports key error metrics.

## Features

- **Upload CSV Dataset:** Easily upload your own data (target variable must be the last column).
- **Regression Models:** Choose between Linear Regression and Lasso Regression.
- **Automated Preprocessing:** Handles missing values and encodes categorical variables.
- **Visualization:** Interactive plot comparing actual vs. predicted values.
- **Error Metrics:** Reports Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/regression-analysis-webapp.git
   cd regression-analysis-webapp
   ```

2. **Install dependencies:**
   ```bash
   pip install flask pandas scikit-learn matplotlib
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**  
   Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Usage

1. Upload a CSV file. The last column should be the target variable.
2. Select the regression model (Linear or Lasso).
3. Click "Analyze" to view error metrics and the regression plot.

## File Structure

```
regression-analysis-webapp/
│
├── app.py
├── README.md
└── templates/
    └── index.html
```



## Notes

- Your dataset should be formatted such that the last column is the target variable (what you want to predict).
- The app will automatically drop rows with missing values and encode categorical features.

## License

MIT License

---

*Built with Flask, Scikit-learn, Pandas, and Matplotlib.*
