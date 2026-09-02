from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "Resume Job Matcher is Working!"


if __name__ == "__main__":
    app.run(debug=True)