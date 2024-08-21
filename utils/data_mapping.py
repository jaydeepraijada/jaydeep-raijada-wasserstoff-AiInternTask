import json
import uuid

def generate_unique_id():
    """Generates a unique ID for each object."""
    return str(uuid.uuid4())

def map_object_data(object_id, description, extracted_text, summary):
    """
    Maps data to a single object.
    
    Args:
        object_id (str): Unique ID of the object.
        description (str): Description of the object.
        extracted_text (list): List of extracted text items.
        summary (str): Summary of the object's attributes.
    
    Returns:
        dict: A dictionary containing mapped data for the object.
    """
    return {
        "object_id": object_id,
        "description": description,
        "extracted_text": extracted_text,
        "summary": summary
    }

def create_summary_table(objects_data):
    """
    Creates a data structure to store all objects' data along with the master image.
    
    Args:
        objects_data (list): List of dictionaries containing data for each object.
    
    Returns:
        dict: A dictionary containing the complete mapping of the master image and its objects.
    """
    master_id = generate_unique_id()
    data_mapping = {
        "master_image_id": master_id,
        "objects": objects_data
    }
    return data_mapping

def save_mapping_to_json(data_mapping, output_path):
    """
    Saves the data mapping to a JSON file.
    
    Args:
        data_mapping (dict): The complete data mapping structure.
        output_path (str): File path to save the JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(data_mapping, json_file, indent=4)
