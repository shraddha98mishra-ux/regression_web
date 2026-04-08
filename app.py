from flask import Flask, render_template, request
from model import run_model, generate_plots

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    coef = []
    mse = None
    r2 = None
    model = "Ridge"
    lam = 1

    if request.method == "POST":
        model = request.form["model"]
        lam = float(request.form["lambda"])

        coef, mse, r2 = run_model(model, lam)
        generate_plots(model)

    return render_template("index.html",
                           coef=coef,
                           mse=mse,
                           r2=r2,
                           model=model,
                           lam=lam,
                           trained=request.method=="POST")

if __name__ == "__main__":
    app.run(debug=True)
