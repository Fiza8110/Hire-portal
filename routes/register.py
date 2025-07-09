from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from config.config import REGISTER_COL
from models.models import User
import httpx

route = APIRouter()

Signup = 'Signup.html'

pwd_encode = CryptContext(schemes=["bcrypt"], deprecated="auto")

templates = Jinja2Templates(directory='templates')

route.mount("/static", StaticFiles(directory="static"), name="static")

# Your Google reCAPTCHA Secret Key
RECAPTCHA_SECRET_KEY = "6LfdEV0rAAAAAJCiH0CBuylmny_MWSzFSFF5X7LC"

# Route to render the Signup HTML page
@route.get("/register")
def register(request: Request): 
    return templates.TemplateResponse(Signup, {"request": request})

# Route to register details
@route.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    g_recaptcha_response: str = Form(alias="g-recaptcha-response")
):
    try:
        # Step 1: Verify CAPTCHA
        async with httpx.AsyncClient() as client:
            captcha_response = await client.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data={
                    "secret": RECAPTCHA_SECRET_KEY,
                    "response": g_recaptcha_response
                }
            )
        captcha_result = captcha_response.json()
        if not captcha_result.get("success"):
            return JSONResponse(content={"msg": "CAPTCHA verification failed. Please try again."}, status_code=400)

        # Step 2: User validation and registration
        user_db = REGISTER_COL.find_one({'email': email}, {'_id': 0})
        if not user_db:
            if username:
                if confirm_password == password:
                    if len(password) >= 7 and any(char.isupper() for char in password) and any(not char.isupper() for char in password):
                        hash_password = pwd_encode.hash(password)
                        user_data = User(username=username, email=email, password=hash_password, role="user")
                        REGISTER_COL.insert_one(dict(user_data))
                        return JSONResponse(content={"success_msg": "Successfully Registered! Please Login"}, status_code=200)
                    return JSONResponse(content={"msg": "Password must contain at least one uppercase & be 7+ characters."}, status_code=401)
                return JSONResponse(content={"msg": "Passwords do not match"}, status_code=401)
            return JSONResponse(content={"msg": "Username is required"}, status_code=401)
        return JSONResponse(content={"msg": "Email already exists"}, status_code=401)

    except ValueError:
        return JSONResponse(content={"msg": "Invalid email format"}, status_code=401)
    except Exception as e:
        return JSONResponse(content={"msg": f"Server error: {str(e)}"}, status_code=500)
