import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess(df):
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if not np.issubdtype(X.dtypes[0], np.number):
        X = pd.get_dummies(X)
    return X, y

def plot_results(y_test, y_pred, title='Regression'):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, c='blue', label='Predictions')
    plt.plot(y_test, y_test, color='red', label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = {}
    img_data = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            X, y = preprocess(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_type = request.form.get('model', 'linear')
            if model_type == 'lasso':
                model = Lasso(alpha=0.1)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics['MSE'] = mean_squared_error(y_test, y_pred)
            metrics['MAE'] = mean_absolute_error(y_test, y_pred)
            metrics['R2'] = r2_score(y_test, y_pred)

            plot_buf = plot_results(y_test, y_pred, title=f"{model_type.capitalize()} Regression")
            img_data = plot_buf.getvalue()
            img_data = 'data:image/png;base64,' + (np.frombuffer(img_data, dtype=np.uint8).tobytes().encode('base64').decode())
    return render_template('index.html', metrics=metrics, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
