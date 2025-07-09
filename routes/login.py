from fastapi import APIRouter, Request,Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from routes.create_token import authenticate_user,create_access_token
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

route = APIRouter()

oauth2_schema = OAuth2PasswordBearer(tokenUrl="token")

templates = Jinja2Templates(directory='templates')

route.mount("/static", StaticFiles(directory = "static"), name = "static")

# Route to render login html page
@route.get("/login")
def login(request: Request): 
    return templates.TemplateResponse("Login.html", {"request": request})

#Route for login details
@route.post("/login")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        # Authenticate user from database
        user = authenticate_user(email=form_data.username, password=form_data.password)
        access_token = create_access_token(user_data={"sub": user["email"]})
        # Store in session
        request.session["user_email"] = user["email"]
        request.session["username"] = user["username"]

        return JSONResponse(
            content={
                "access_token": access_token,
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
            },
            status_code=200
        )

      
    except Exception as e:
        return JSONResponse(content={"msg": f"Error: {str(e)}"}, status_code=500)
    
