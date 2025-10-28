from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    data = yf.download(symbol, period='6mo', interval='1d')
    
    data['Price_Change'] = data['Close'] - data['Open']
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)

    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
    latest = X.tail(1)
    prediction = model.predict(latest)[0]

    result = "📈 Stock likely to go UP tomorrow" if prediction == 1 else "📉 Stock likely to go DOWN tomorrow"
    return render_template('index.html', result=result, accuracy=acc, symbol=symbol.upper())

if __name__ == '__main__':
    app.run(debug=True)
