from flask import Flask, jsonify, request, render_template
from model import predict


app = Flask('app')

@app.route("/")
def hello():
    return render_template('index.html')
    
  
@app.route("/generate", methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        start_string = request.form['startString']
        start_string = start_string.lower()
        start_string = start_string + ' ' 
        x = predict(start_string)
        return render_template('results.html', prediction = x)
    elif request.method == 'POST':
        start_string = request.form['startString']
        start_string = start_string.lower()
        start_string = start_string + ' ' 
        x = predict(start_string)
        return render_template('results.html', prediction = x)
    else:
        start_string = request.form['startString']
        start_string = start_string.lower()
        start_string = start_string + ' ' 
        x = predict(start_string)
        return render_template('results.html', prediction = x)

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)