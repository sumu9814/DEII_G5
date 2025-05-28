from celery import Celery
import subprocess
import json

# connect to RabbitMq
celery = Celery('workerA', broker='amqp://rabbitmq:rabbitmq@rabbit:5672/', backend='rpc://')

# define task
@celery.task(name="workerA.get_predictions")
def get_predictions(csv_string):
    try:
        # send data to prediction script
        result = subprocess.run(
            ["python3", "/app/app_files/predict.py"],
            input=csv_string.encode("utf-8"),
            capture_output=True,
            check=True
        )

        # send back result to be displayed
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": f"Subprocess failed: {e.stderr.decode()}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode failed: {str(e)}"}
