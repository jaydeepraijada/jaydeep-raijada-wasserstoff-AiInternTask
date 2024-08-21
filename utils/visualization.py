import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
from io import BytesIO

def generate_annotated_image(image, masks, threshold=0.5):
    """
    Generate an annotated image with masks overlaid.
    
    Parameters:
    - image (PIL.Image.Image): The original image.
    - masks (list of dict): List of masks with their respective scores.
    - threshold (float): Minimum score to display a mask.

    Returns:
    - PIL.Image.Image: Annotated image.
    """
    fig, ax = plt.subplots()
    ax.imshow(np.array(image))

    for mask in masks:
        if mask['score'] > threshold:
            mask_arr = mask['mask'].squeeze().astype(np.uint8)
            ax.imshow(mask_arr, cmap='jet', alpha=0.5)  # Overlay mask on image

    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    annotated_image = Image.open(buf)
    buf.close()
    
    return annotated_image

def save_annotated_image(image, output_path):
    """
    Save the annotated image to a file.
    
    Parameters:
    - image (PIL.Image.Image): The annotated image.
    - output_path (str): Path where the image will be saved.
    """
    image.save(output_path)

def create_summary_table(objects_data):
    """
    Create a summary table from the objects data.
    
    Parameters:
    - objects_data (list of dict): List containing data for each object.
    
    Returns:
    - pandas.DataFrame: Summary table.
    """
    df = pd.DataFrame(objects_data)
    return df

def save_summary_table(df, output_path):
    """
    Save the summary table to a CSV file.
    
    Parameters:
    - df (pandas.DataFrame): Summary table.
    - output_path (str): Path where the table will be saved.
    """
    df.to_csv(output_path, index=False)

def generate_output(image, masks, objects_data, master_id, output_dir="data/output"):
    """
    Generate and save the final output including annotated image and summary table.
    
    Parameters:
    - image (PIL.Image.Image): The original image.
    - masks (list of dict): List of masks with their respective scores.
    - objects_data (list of dict): List of data for each object.
    - master_id (str): Unique identifier for the master image.
    - output_dir (str): Directory to save the output files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate annotated image
    annotated_image = generate_annotated_image(image, masks)
    annotated_image_path = os.path.join(output_dir, f"{master_id}_annotated.png")
    save_annotated_image(annotated_image, annotated_image_path)
    
    # Create and save summary table
    summary_table = create_summary_table(objects_data)
    summary_table_path = os.path.join(output_dir, f"{master_id}_summary.csv")
    save_summary_table(summary_table, summary_table_path)
    
    return annotated_image_path, summary_table_path
