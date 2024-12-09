import os
import pickle


from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

with open("nb_model.pkl", "rb") as model_file:
    nb_classifier = pickle.load(model_file)


@app.route("/", methods=["GET", "POST"])
def index_page():
    prediction = ""
    if request.method == "POST":
        level = request.form["level"]
        lang = request.form["lang"]
        tweets = request.form["tweets"]
        phd = request.form["phd"]
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction)


@app.route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "unknown")
    return "<h1>Hello, {}!</h1>".format(name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
