from flask import Flask, request, render_template
from app_files.workerA import get_predictions
import time
import pandas as pd

app = Flask(__name__)

# main page -  prediction page
@app.route("/")
def index():
    return render_template("upload.html")

# predict page - upload csv file for predictions
@app.route("/predict", methods=["POST"])
def trigger_prediction():
    try:
        # prompt to upload
        if "csv_file" not in request.files:
            return "Please upload file", 400

        # check if submission has csv file
        csv = request.files["csv_file"]
        if csv.filename == "":
            return "No file detected", 400

        # read data from csv
        csv_data = csv.read().decode("utf-8")
        start_time = time.time()

        # send task to celery worker
        taskA = get_predictions.delay(csv_data)
        res_predictions = taskA.get(timeout=120)
        
        # save time for analysis
        total_time = time.time() - start_time

        #sort repositories by predictions
        if isinstance(res_predictions, list) and "predicted_stars" in res_predictions[0]:
            res_predictions = sorted(res_predictions, key=lambda x: x["predicted_stars"], reverse=True)
            res_predictions = res_predictions[:5]

        # if task succesful - show results page
        return render_template("result.html", predictions=res_predictions, duration=f"{total_time:.2f}")
    except Exception as e:
        return f"Error: {str(e)}", 500
    

# page randering for form
@app.route("/manual")
def manual_entry():
    return render_template("form.html")

# submit method for form
@app.route("/submit_manual", methods=["POST"])
def submit_manual():
    try:

        # create requested fields
        data_repo = {
            "description_len": int(request.form["description_len"]),
            "language": request.form["language"],
            "has_issues": int(request.form.get("has_issues", 0)),
            "has_wiki": int(request.form.get("has_wiki", 0)),
            "days_since_created": int(request.form["days_since_created"]),
            "days_since_updated": int(request.form["days_since_updated"]),
            "days_since_pushed": int(request.form["days_since_pushed"]),
            "commit_count": int(request.form["commit_count"])
        }

        # convert data to csv
        df = pd.DataFrame([data_repo])
        csv = df.to_csv(index=False)

        # start time counting for task
        start_time = time.time()

        # send task to celery worker
        taskA = get_predictions.delay(csv)
        repo_result = taskA.get(timeout=120)

        # get total time for analysis
        total_time = time.time() - start_time

        # if no error show result page
        return render_template("result.html", predictions=repo_result, duration=f"{total_time:.2f}")
    except Exception as e:
        return f"Error: {str(e)}", 500


# app port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)
