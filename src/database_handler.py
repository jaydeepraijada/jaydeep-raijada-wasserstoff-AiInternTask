import os
import json
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

client = None
db = None
collection = None

def init_mongodb():
    global client, db, collection
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pdf_database']
        collection = db['pdf_documents']
        print("Successfully connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        raise

def close_mongodb():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")

async def update_mongodb(file_path, summary, keywords):
    global collection
    if collection is None:
        init_mongodb()
    file_name = os.path.basename(file_path)
    document = {
        'file_name': file_name,
        'file_path': file_path,
        'summary': summary,
        'keywords': keywords
    }
    json_document = json.dumps(document)
    try:
        result = await collection.update_one(
            {'file_name': file_name},
            {'$set': json.loads(json_document)},
            upsert=True
        )
        return result
    except Exception as e:
        logger.error(f"Error updating MongoDB for {file_name}: {str(e)}")

def check_mongodb_connection():
    try:
        client.server_info()
        print("Successfully connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        return False
    return True