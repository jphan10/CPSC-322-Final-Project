import os
import pickle


from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

with open("models/nb_model.pkl", "rb") as model_file:
    nb_classifier = pickle.load(model_file)


@app.route("/", methods=["GET", "POST"])
def index_page():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    prediction = ""
    if request.method == "POST":
        elo1_pre = request.form["elo1_pre"]
        elo2_pre = request.form["elo2_pre"]
        elo_prob1 = request.form["elo_prob1"]
        elo_prob2 = request.form["elo_prob2"]
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        score1 = request.form["score1"]
        score2 = request.form["score2"]

        # Assuming the model expects a list of features
        features = [
            elo1_pre,
            elo2_pre,
            elo_prob1,
            elo_prob2,
            team1,
            team2,
            score1,
            score2,
        ]
        prediction = nb_classifier.predict([features])[0]
    print("prediction:", prediction)
    return render_template("index.html", prediction=prediction)


@app.route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "unknown")
    return "<h1>Hello, {}!</h1>".format(name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
