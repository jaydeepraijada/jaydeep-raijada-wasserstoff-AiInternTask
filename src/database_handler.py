import logging
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self, client: AsyncIOMotorClient, database_name: str, collection_name: str):
        self.client = client
        self.db = client[database_name]
        self.collection = self.db[collection_name]

    async def insert_document(self, document):
        try:
            result = await self.collection.insert_one(document)
            logger.info(f"Document inserted with id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            raise

    async def update_document(self, pdf_file, summary, keywords):
        try:
            result = await self.collection.update_one(
                {"pdf_file": pdf_file},
                {"$set": {"summary": summary, "keywords": keywords}},
                upsert=True
            )
            if result.upserted_id:
                logger.info(f"Document inserted with id: {result.upserted_id}")
            else:
                logger.info(f"Document updated: {result.modified_count} modified")
            return result
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    async def close(self):
        self.client.close()
