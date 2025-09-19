# reference: https://www.learncodewithmike.com/2020/02/python-email.html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
import sys
import requests

# email
SMTP_SERVER_EMAIL_ADDRESS = "" # your server mail address
SMTP_SERVER_GMAIL_APP_PASSWORD = "" # your gmail application password
CLIENT_EMAIL_ADDRESS = "" # your client mail address

def model_training_complete(task_name, server_name):
    content = MIMEMultipart()  # create MIMEMultipart object
    content["subject"] = f"{task_name} completed"
    # SMTP server email address
    content["from"] = SMTP_SERVER_EMAIL_ADDRESS
    # client email address
    content["to"] = CLIENT_EMAIL_ADDRESS

    # text = f"""Hello {(CLIENT_EMAIL_ADDRESS.split('@')[0])},
    # The {task_name} has been completed in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, please check the result in server {server_name}
    # Good Luck!"""

#     content.attach(MIMEText(text, 'plain'))

    html = f"""
    <html>
    <head></head>
    <body>
        <p>
            Hello {(CLIENT_EMAIL_ADDRESS.split('@')[0])},<br><br>
            &nbsp;&nbsp;&nbsp;&nbsp;The <b>{task_name}</b> has been completed in <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>, please check the result in server <b>{server_name}</b>. <br><br>
            Good Luck!<br>
            <a href="https://media.tenor.com/AVruRQee2NoAAAAd/puppy-dance-dog-puppies.gif" target="_blank">
                <img src="https://media.tenor.com/AVruRQee2NoAAAAd/puppy-dance-dog-puppies.gif" border="0" alt="dancing puppy" width="150">
            </a>
    </body>
    </html>
    """

    content.attach(MIMEText(html, 'html'))

    return content

def model_inference_complete(task_name, server_name):
    content = MIMEMultipart()  # create MIMEMultipart object
    content["subject"] = f"Inference model {task_name} completed"
    # SMTP server email address
    content["from"] = SMTP_SERVER_EMAIL_ADDRESS
    # client email address
    content["to"] = CLIENT_EMAIL_ADDRESS

    # text = f"""Hello {(CLIENT_EMAIL_ADDRESS.split('@')[0])},
    # The {task_name} has been completed in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, please check the result in server {server_name}
    # Good Luck!"""

#     content.attach(MIMEText(text, 'plain'))

    html = f"""
    <html>
    <head></head>
    <body>
        <p>
            Hello {(CLIENT_EMAIL_ADDRESS.split('@')[0])},<br><br>
            &nbsp;&nbsp;&nbsp;&nbsp;Inference all epoch for model: <b>{task_name}</b> has been completed in <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>, please check the result in server <b>{server_name}</b>. <br><br>
            Good Luck!<br>
            <a href="https://media.tenor.com/AVruRQee2NoAAAAd/puppy-dance-dog-puppies.gif" target="_blank">
                <img src="https://media.tenor.com/AVruRQee2NoAAAAd/puppy-dance-dog-puppies.gif" border="0" alt="dancing puppy" width="150">
            </a>
    </body>
    </html>
    """

    content.attach(MIMEText(html, 'html'))

    return content


def send_email(content):
    # setting SMTP server
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:
        try:
            # verify SMTP server
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_SERVER_EMAIL_ADDRESS,
                       SMTP_SERVER_GMAIL_APP_PASSWORD)
            smtp.send_message(content)
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)

if __name__ == "__main__":
    task_name = "training model"
    server_name = "121-nasic01"
    
    if len(sys.argv) == 2:
        # pass: task_name
        task_name = sys.argv[1]
    elif len(sys.argv) == 3:
        # pass: task_name, server_name
        task_name = sys.argv[1]
        server_name = sys.argv[2]

    print("Send email!")
    content = model_training_complete(task_name = task_name, server_name = server_name)
    # content = model_inference_complete(task_name = task_name, server_name = server_name)
    send_email(content)
