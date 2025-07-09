from fastapi import APIRouter,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
import random
from datetime import datetime, timedelta
from config.config import APPLICATION_COL
from pydantic import BaseModel
from dotenv import load_dotenv  
import os

# Load environment variables from .env file
load_dotenv()

route = APIRouter()

templates = Jinja2Templates(directory='templates')

route.mount("/static", StaticFiles(directory = "static"), name = "static")

contactus = "contactus.html"

# Route to render home html page
@route.get("/") # / Default path
def home(request: Request): 
    return templates.TemplateResponse("Signup.html", {"request": request})


user_data = {}
# Email settings
MAIL_USERNAME=os.getenv("EMAIL")

class ScheduleRequest(BaseModel):
    action: str
    to: str
    date: str = None  # Optional, only for scheduling
    time: str = None  # Optional, only for scheduling
    reason: str = None  # Optional, only for rejection


def send_email(receiver_email, email_content,subject):
    sender_email = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PASSWORD")  
    # subject = "Scheduled Interview"

    # Construct the email message
    message = f"""
    {email_content}

    Best Regards,
    XYZ Company
    """

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    # Send email using Gmail SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        print(f"Email successfully sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")
def send_email1(receiver_email, email_content):
    sender_email = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PASSWORD") 
    subject = "Reset The Password"

    # Construct the email message
    message = f"""
    Dear User,

    We received a request to reset your password for your account at XYZ Company.
    {email_content}

    If you did not request this change, please ignore this email. Otherwise, please follow the instructions above to reset your password.

    Best Regards,  
    XYZ Company Support Team
    """

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    # Send email using Gmail SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        print(f"Email successfully sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

def send_application_success_email(receiver_email, name, job_title):
    sender_email = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PASSWORD") 

    subject = "Job Application Received"
    message = f"""
    Dear {name},
    Thank you for applying for the position of {job_title} at our company.
    We have received your application successfully.

    Our recruitment team will review your submission and get back to you soon.

    Best regards,  
    HR Team
    Exafluence
    """

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        print(f" Success email sent to {receiver_email}")
    except Exception as e:
        print(f" Error sending email to {receiver_email}: {e}")

        
@route.post("/scheduleInterview")
async def schedule_interview(request: ScheduleRequest):
    # Extract data from request
    action = request.action
    mail = request.to
    date = request.date
    time = request.time
    reason = request.reason

    # Print action (button ID)
    print(f"Action: {action}")

    if action == "scheduleInterviewBtn":
        # Send email for interview scheduling
        email_content = f"""
        Dear Candidate,

        We are pleased to inform you that your interview has been scheduled as per the details below:
        📅 Date: {date} 
        ⏰ Time: {time}  
        🔗 Google Meet Link: https://meet.google.com/sample-link
        
        Please ensure you are available at least 10 minutes before the scheduled time and that you have a stable internet connection. The interview will be conducted virtually, so kindly join using the provided link.
        
        If you have any questions or concerns prior to the interview, feel free to reach out to us.

        Best of luck!
        Warm regards,  
        HR Team
        """
        send_email(mail, email_content,"Schedule Interview")

    elif action == "rejectApplicationBtn":
        # Update application status to Rejected
        APPLICATION_COL.update_one(
           
            {"email": mail},
            {"$set": {
                "status": "Rejected",
                "rejectionReason": reason
            }}
        )
        print(f"Candidate with email {mail} has been marked as Rejected for the following reason: {reason}")
        # Send rejection email
        email_content = f"""
        Dear Candidate,

        Thank you for taking the time to apply for the position at our company. We appreciate your interest and the effort you put into your application.

        After careful consideration, we regret to inform you that your application has not been successful at this time. The reason for rejection is as follows:

        Reason: {reason}

        Please do not be discouraged, as this decision does not reflect your overall potential or abilities. We encourage you to apply for future opportunities that match your skills and experience.

        We wish you all the best in your job search and future career endeavors.

        Kind regards,  
        HR Team
        """
        send_email(mail, email_content,"Rejection")

    # Return success response
    response_data = {"message": "Mail sent successfully"}
    return JSONResponse(status_code=200, content=response_data)
