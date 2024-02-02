from flask import Flask, request, jsonify, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Sentiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    sentiment = db.Column(db.Integer, nullable=False)

# Moved db.create_all() inside a function
def setup_database(app):
    with app.app_context():
        db.create_all()

tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("kakaobank/kf-deberta-base")

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = scores.argmax().item() - 1  # Adjusting to {-1, 0, 1} scale
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sentiment = analyze_sentiment(text)
    sentiment_record = Sentiment(text=text, sentiment=sentiment)
    db.session.add(sentiment_record)
    db.session.commit()
    return jsonify({'sentiment': sentiment})

@app.route('/records')
def records():
    sentiments = Sentiment.query.all()  # Fetch all sentiment records from the database
    return render_template('records.html', sentiments=sentiments)

@app.route('/download')
def download():
    sentiments = Sentiment.query.all()
    csv_file = 'sentiments.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Input Text', 'Output Score'])
        for sentiment in sentiments:
            writer.writerow([sentiment.text, sentiment.sentiment])

    return_data = open(csv_file, 'r').read()
    response = make_response(return_data)
    response.headers['Content-Disposition'] = 'attachment; filename=sentiments.csv'
    response.mimetype = 'text/csv'
    return response

if __name__ == '__main__':
    setup_database(app)  # Call the function to setup the database
    app.run(debug=True)
