import os
import subprocess


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL_AUTOMATIC_DETECT")

def send_slack_notification(message):
    hostname = subprocess.run(["hostname"], stdout=subprocess.PIPE, text=True).stdout.strip()
    full_message = f"[{hostname}] {message}"
    subprocess.run([
        "curl", "-X", "POST", "-H", "Content-type: application/json",
        "--data", f'{{"text":"{full_message}"}}', SLACK_WEBHOOK_URL
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)