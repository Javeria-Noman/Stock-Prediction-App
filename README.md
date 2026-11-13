# Stock Price Prediction Web Application

A full-stack web application built with **Flask** that fetches real-time stock market data, performs exploratory data analysis (EDA), and predicts stock trends using machine learning and data science libraries. The frontend is styled using a modern combination of **HTML5, CSS3, Tailwind CSS, and Bootstrap**.

---

## Features

- **Real-Time Data Fetching**: Downloads historical and current stock data dynamically using `yfinance`.
- **Interactive Web Interface**: Clean, responsive UI designed with Tailwind CSS and Bootstrap.
- **Data Visualization**: Generates trend charts (Closing prices, Open vs. Close, High vs. Low) using Matplotlib and Seaborn.
- **Flask Backend**: Lightweight and fast Python web server handling routing and data processing.
- **Modular Structure**: Organized directories for data storage, trained models, notebooks, and generated outputs.

---

## Tech Stack

- **Backend Framework**: Flask (Python)
- **Frontend**: HTML5, CSS3, Tailwind CSS, Bootstrap
- **Data Science & ML**: Pandas, NumPy, Scikit-Learn, TensorFlow, Seaborn, Matplotlib[cite: 1]
- **API**: Yahoo Finance (`yfinance`)[cite: 1]

---

## 📁 Project Structure

The project follows a structured directory layout[cite: 1]:

```text
stock_price_prediction/
│
├── data/                  # Downloaded stock datasets (e.g., CSV files)[cite: 1]
├── notebooks/             # Jupyter / Google Colab notebooks for EDA & modeling[cite: 1]
├── models/                # Saved machine learning / deep learning models[cite: 1]
├── outputs/               # Generated graphs and visual plots (.png)[cite: 1]
├── static/                # CSS, JavaScript, and image assets




├── templates/             # HTML files for Flask rendering
├── app.py                 # Main Flask application file
└── requirements.txt       # Project dependencies[cite: 1]
