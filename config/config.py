import os
import pymongo
from gridfs import GridFS
from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv()

# Establishing MongoDB server
MONGO_URI = os.getenv("MONGO_URI")  # Load from .env

# Instance for MongoDB server
Client = pymongo.MongoClient(MONGO_URI)

# Creating DB
DB = Client["HrPortal"]

# Creating users collection to store the user credentials
REGISTER_COL = DB["USERS"]

JObs_COL = DB["Jobs"]
APPLICATION_COL = DB["Applications"]
fs = GridFS(DB)