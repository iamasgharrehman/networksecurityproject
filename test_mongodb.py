import os
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve MongoDB connection URL from .env file
MONGODB_URL = os.getenv("MONGODB_URL")

# Ensure the URL is retrieved
if not MONGODB_URL:
    raise ValueError("MONGODB_URL is not set in the environment variables.")

# Create a new client and connect to the server
client = MongoClient(MONGODB_URL)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
