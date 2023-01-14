from flask import Flask , render_template , request
import utils
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("sarcasm.pkl","rb"))
s = pickle.load(open("count_vector.pkl","rb"))

@app.route('/')
def home():
    return render_template("index_1.html")

@app.route("/predict/",methods =["GET","POST"])

def predict():
    if request.method == "POST":
        sen = str(request.form.get(("sen")))
        #print(sen)
        X = s.transform([sen]).toarray()
        #print(s)
        output = model.predict(X)
        #print(output)
    return render_template("predict.html",prediction=output)

if __name__ == "__main__":
    app.run(debug=True)
