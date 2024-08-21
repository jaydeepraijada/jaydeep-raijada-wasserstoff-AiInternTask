import numpy as np
import cv2
import PIL as Image
import json
import os
import uuid



#Object Exraction

def extract_object(image, mask):
    img_np = np.array(image)
    
    # Resize mask to match image dimensions
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create an empty image with the same dimensions as the original image
    object_img = np.zeros_like(img_np)

    # Apply the mask to the image
    for c in range(3):  # Assuming image has 3 channels (RGB)
        object_img[:, :, c] = img_np[:, :, c] * mask_resized
    
    return Image.fromarray(object_img)


# Save the input image
def save_input_image(image, master_id):
    input_images_dir = 'data/input_images/'
    
    os.makedirs(input_images_dir, exist_ok=True)
    
    input_image_path = os.path.join(input_images_dir, f'{master_id}.png')
    image.save(input_image_path)
    return input_image_path

# Save the extracted objects and their metadata
def save_objects_and_metadata(extracted_objects, master_id):
    segmented_objects_dir = 'data/segmented_objects/'
    os.makedirs(segmented_objects_dir, exist_ok=True)

    object_metadata = []
    
    for i, obj_img in enumerate(extracted_objects):
        object_id = str(uuid.uuid4())
        object_image_path = os.path.join(segmented_objects_dir, f'{object_id}.png')
        obj_img.save(object_image_path)
        
        metadata = {
            'object_id': object_id,
            'master_id': master_id,
            'object_image_path': object_image_path
        }
        object_metadata.append(metadata)
    
    metadata_file = os.path.join(segmented_objects_dir, f'{master_id}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(object_metadata, f, indent=4)
    
    return object_metadata
