from flask import Flask, request, render_template
import pandas as pd
from predictor import predict_ticket

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticket_text = request.form.get('ticket', '')
    if not ticket_text.strip():
        return render_template('index.html',
                               error="Please enter a ticket description.")
    result = predict_ticket(ticket_text)
    return render_template('index.html',
                           ticket=ticket_text,
                           result=result)

@app.route('/bulk', methods=['GET', 'POST'])
def bulk():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('bulk.html',
                                   error="Please upload a CSV file.")
        df = pd.read_csv(file)

        # Auto detect text column
        text_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['narrative', 'complaint', 'text', 'description']):
                text_col = col
                break
        if not text_col:
            text_col = df.columns[0]

        # Predict
        df['Category'] = df[text_col].apply(
            lambda x: predict_ticket(str(x))['category']
        )
        df['Priority'] = df[text_col].apply(
            lambda x: predict_ticket(str(x))['priority']
        )

        total           = len(df)
        priority_counts = df['Priority'].value_counts().to_dict()
        category_counts = df['Category'].value_counts().to_dict()
        table_html      = df.head(20).to_html(classes='table', index=False)

        return render_template('bulk.html',
                               total=total,
                               priority_counts=priority_counts,
                               category_counts=category_counts,
                               table_html=table_html)

    return render_template('bulk.html')

if __name__ == '__main__':
    app.run(debug=True)