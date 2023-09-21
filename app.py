from flask import Flask, render_template, request, jsonify
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mine_rules', methods=['POST'])
def mine_rules():
    try:
        # Get the CSV file path from the HTML form
        csv_file_path = request.form['csv_file_path']
        
        # Read the CSV data
        df = pd.read_csv(csv_file_path, names=['products'], sep=',')
        data = list(df["products"].apply(lambda x: x.split(",")))

        # Perform Apriori analysis
        a = TransactionEncoder()
        a_data = a.fit(data).transform(data)
        df = pd.DataFrame(a_data, columns=a.columns_)
        df = df.replace(False, 0)
        df = apriori(df, min_support=0.2, use_colnames=True, verbose=1)
        df_ar = association_rules(df, metric="confidence", min_threshold=0.6)

        # Convert the result to a JSON response
        result = df_ar.to_dict(orient='records')
        
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

