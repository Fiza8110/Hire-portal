from pydantic import BaseModel,EmailStr

#User pydantic models
class User(BaseModel):
    username : str
    email : EmailStr
    password : str
    role : str
