import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import uuid

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.segmentation_model import load_model, run_inference 
from models.identification_model import load_yolov8_model, run_object_detection
from models.text_extraction_model import extract_text
from models.summarization_model import generate_description, summarize_text_and_image

from utils.preprocessing import transform_image
from utils.postprocessing import save_input_image, save_objects_and_metadata, extract_object
from utils.data_mapping import map_object_data, create_summary_table, save_mapping_to_json
from utils.visualization import generate_output


#loading the required models
model = load_model() #segmentation_model
detection_model = load_yolov8_model()

# def resize_image(image, size=(800, 800)):
#     return image.resize(size, Image.ANTIALIAS)

def display_masks(outputs, image, threshold=0.5):
    masks = outputs[0]['masks'] #takes the outputs from output of segmentation model(masks)
    scores = outputs[0]['scores'] #scores
    
    fig, ax = plt.subplots() #creating a subplot for showing the original image
    ax.imshow(np.array(image)) #show original image

    extracted_objects = []
    
    for i in range(len(scores)):
        if scores[i] > threshold: #if score of extracted image is above threshold

            mask = masks[i].squeeze().cpu().numpy() #squeezes mask(removes dimensions of size 1), moves to cpu, converts to np array
        
            mask = np.where(mask > 0.5, 1, 0).astype(np.uint8) #values above 0.5 to 1, rest all to 0

            object_img = extract_object(image,mask) #returns image
            extracted_objects.append(object_img)
            #Display the mask
            ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask on original image
    
    st.pyplot(fig) #show segmented original image

    return extracted_objects

st.title("AI Pipeline: Image Segmentation, Object Detection, and Text Extraction")
st.sidebar.header("Options")
st.sidebar.text("Upload an image to start processing.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = uploaded_file
    st.image(image, caption='Uploaded Image.', use_column_width=True) #show uploaded image
    image = Image.open(uploaded_file).convert('RGB') 

    # Generate a unique master ID for the image
    master_id = str(uuid.uuid4())

    st.header("Results") #header for results

    description = generate_description(image) #RGB image
    st.write("Generated Description:", description)
    
    #extract text for the entire image
    extracted_text = extract_text(image) 
    #Input is RGB image - converted to np.array , returns extracted text

    st.subheader("Extracted Text")
    if extracted_text:
        st.write(extracted_text)
    else:
        st.write("No text was detected")

    #summarize the entire image
    summary = summarize_text_and_image(description, extracted_text)
    st.subheader("Image Summary")
    st.write("Generated Summary:", summary)

    # Save the input image
    save_input_image(image, master_id)

    # Transform image
    image_tensor = transform_image(image) #resize (256,256), to_tensor, normalize
    outputs = run_inference(model, image_tensor) #outputs are masks, labels, scores , bb

    
    extracted_objects = display_masks(outputs, image) 
    #saves extracted object to extracted_objects, show segmented image

    objects_data = []

    if extracted_objects: #if objects are found
    # Save the extracted objects and their metadata
        metadata = save_objects_and_metadata(extracted_objects, master_id)
    
    
    # Display each extracted object
        st.write("Extracted Objects:")
        for i, obj_img in enumerate(extracted_objects):
         st.image(obj_img, caption=f'Object {i+1}', use_column_width=True)

         obj_description = generate_description(obj_img)
         st.write("Generated Description:", description)

         # Convert the object image to a numpy array for YOLO inference
         obj_img_np = np.array(obj_img)

         # Run object detection on each extracted object
         detection_results = run_object_detection(detection_model, obj_img_np)
         st.write(f"Detection results for Object {i+1}:")
         st.json(detection_results)
         
         obj_text = extract_text(obj_img)
         if obj_text:
            st.write(f"Extracted Text for Object {i+1}:")
            st.json(obj_text)
         else:
            st.write("No text was detected")

         obj_summary = summarize_text_and_image(obj_description, obj_text)
         st.write(f"Object Summary:\n{obj_summary}")

         object_id = str(uuid.uuid4())
         object_data = map_object_data(object_id, obj_description, obj_text, obj_summary)
         objects_data.append(object_data)

        data_mapping = create_summary_table(objects_data)
        output_path = os.path.join("data", "output", f"{master_id}_data_mapping.json")
        save_mapping_to_json(data_mapping, output_path)

        # Generate the final output image with annotations and summary table
        annotated_image_path, summary_table_path = generate_output(image, outputs[0]['masks'], objects_data, master_id)

        st.subheader("Final Output")

        # Display the annotated image
        st.image(annotated_image_path, caption='Annotated Image', use_column_width=True)

        # Provide a download link for the summary table
        st.write("Summary Table:")
        st.write(f"Download the summary table [here](data/output/{master_id}_summary.csv)")

        # Display the mapped data
        st.write("Mapped Data:")
        st.json(data_mapping)

        # # Display the JSON data
        # st.write("Mapped Data:")
        # st.json(data_mapping)
    
    else:
        st.write("No objects were detected")


    