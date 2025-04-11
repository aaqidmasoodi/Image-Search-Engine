from flask import Flask, render_template, request
from search import SearchEngine
import os

app = Flask(__name__)
engine = SearchEngine()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        model = request.form.get('model', '1')
        filetype = request.form.get('filetype', '')
        
        if filetype:
            query = f"{query} filetype:{filetype}"
        
        results = engine.search(query, model=model)
        return render_template('results.html', 
                            query=query,
                            results=results,
                            documents=engine.index['documents'])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)