from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from  routes.create_token  import generate_reset_token, verify_reset_token, hash_password
from config.config import REGISTER_COL
from routes.home import send_email1
from fastapi.templating import Jinja2Templates

# Defines a group of related routes for password reset.
reset_router = APIRouter()
templates = Jinja2Templates(directory="templates")

#Renders the form where users can enter their email to reset their password.
@reset_router.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_form(request: Request):
    return templates.TemplateResponse("ForgotPassword.html", {"request": request})

#It is triggered when a user submits their email to request a password reset.
@reset_router.post("/forgot-password")
def send_reset_link(request: Request, email: str = Form(...)):
    user = REGISTER_COL.find_one({"email": email})#Checks if user exists in the DB.

    if not user:#If no user is found, it returns a 404 JSON error response.
        return JSONResponse(status_code=404, content={"message": "User not found"})
    
    token = generate_reset_token(email) #If user exists, generates a secure token tied to their email.
    reset_link = f"http://localhost:8000/reset-password?token={token}"#Sends an email with a reset link
    email_content = f"Click the link to reset your password: {reset_link}"#Responds with a success or error message.
    send_email1(email, email_content)
    return JSONResponse(status_code=200, content={"message": "Reset email sent"})

@reset_router.get("/reset-password", response_class=HTMLResponse)
# Renders a form for entering a new password, using the provided token
def reset_password_form(request: Request, token: str):
    return templates.TemplateResponse("Forgotpasswordpage.html", {"request": request, "token": token})

@reset_router.post("/reset-password")
def reset_password(request: Request, token: str = Form(...), new_password: str = Form(...)):
    email = verify_reset_token(token)#Verifies the token and retrieves the associated email.
    if not email:#If no email is found, it returns a 404 JSON error response.
        return JSONResponse(status_code=400, content={"message": "Invalid or expired token"})
    
    hashed_pwd = hash_password(new_password)#If token is valid, hashes the new password.
    result = REGISTER_COL.update_one({"email": email}, {"$set": {"password": hashed_pwd}})#Updates the user's password in the database.
    return JSONResponse(status_code=200, content={"message": "Password reset successfully"})#Sends a success response.
