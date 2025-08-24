from flask import Blueprint, request, jsonify
from models.inference_result_model import (
    create_inference_result,
    get_inference_result_by_id,
    get_all_inference_results
)
from models.ml_model import get_weight_file
from middleware.middleware import auth_required
import boto3
import os
import uuid
from dotenv import load_dotenv
from mimetypes import guess_type
import json
import requests
from io import BytesIO
# Load env vars if not already loaded
load_dotenv()

inference_bp = Blueprint('inference_bp', __name__)
INDIVIDUAL_YOLOS_API = os.getenv("INDIVIDUAL_YOLOS_API")
# --- S3 Upload Function ---
def upload_to_s3(file_stream, original_filename, folder='input-images'):
    """
    Uploads a file stream to S3 in the specified folder and returns the public URL.
    """
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    REGION = os.getenv("AWS_REGION", "ap-south-1")
    ENDPOINT = os.getenv("S3_ENDPOINT")
    BUCKET = os.getenv("S3_BUCKET_NAME")
    BUCKET_REGION_URL = os.getenv("S3_BUCKET_REGION_URL")
    

    file_ext = original_filename.rsplit('.', 1)[-1].lower()
    unique_filename = f"yolo-inferences/{folder}/{uuid.uuid4().hex}.{file_ext}"
    content_type = guess_type(original_filename)[0] or "application/octet-stream"

    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        endpoint_url=ENDPOINT
    )

    try:
        s3.upload_fileobj(
            Fileobj=file_stream,
            Bucket=BUCKET,
            Key=unique_filename,
            ExtraArgs={
                "ACL": "public-read",
                "ContentType": content_type
            }
        )
        file_url = f"https://{BUCKET}.{BUCKET_REGION_URL}/{unique_filename}"
        return file_url
    except Exception as e:
        raise RuntimeError(f"S3 Upload failed: {str(e)}")




def run_inference(model_id, image_file, user_email):
    """
    Store the image in the S3 bucket, make the inference via API call,
    and create an inference result record. The output annotated image is also stored in the S3 bucket.
    """
    # Read the file content once
    image_content = image_file.read()
    # Upload image to S3
    try:
        # Rewind original stream so it can be used again (if needed)
        image_file.stream.seek(0)
        image_url = upload_to_s3(image_file, image_file.filename, model_id)
    except RuntimeError as e:
        return 500, {"message": str(e)}

    # Fetch the weight file URL dynamically based on the model_id
    weight_file = get_weight_file(model_id)
    if weight_file[0] != 200:
        return weight_file
    weight_file_url = weight_file[1]['weight_file']
    
    model_name = weight_file[1]['name']
    print("Weight file URL: ", weight_file_url)
    print("Image URL: ", image_url)


    files = {
    'file': (image_file.filename, image_content, image_file.mimetype)
}
    data = {
        'weights': weight_file_url
    }
    # print("Files to be sent:", files)
    print("Data to be sent:", data)

    # Make the request to the YOLO API endpoint
    try:
        response = requests.post(f"{INDIVIDUAL_YOLOS_API}/infer", files=files, data=data)
        if response.status_code != 200:
            return response.status_code, {"message": "Inference failed", "details": response.json()}
        
        data = response.json()  # Assuming the response contains the URL of the annotated image
        print("Inference Response Data:", data)

        # Assuming YOLO API returns the annotated image as binary content or a URL
        annotated_image_url = data.get('image_url')  # This URL would be returned by the YOLO API
        labels_json = data.get('labels', {})  # Assuming labels are returned in the response
        if not annotated_image_url:
            return 500, {"message": "Annotated image URL not returned by the YOLO API"}

        # Fetch the annotated image (if URL is returned)
        annotated_image_response = requests.get(f'{INDIVIDUAL_YOLOS_API}/{annotated_image_url}')
        if annotated_image_response.status_code != 200:
            return 500, {"message": "Failed to fetch annotated image from YOLO API"}

        # Upload the annotated image to S3
        annotated_image_filename = os.path.basename(annotated_image_url)
        annotated_image_url = upload_to_s3(
            BytesIO(annotated_image_response.content),
            annotated_image_filename,
            folder=f'{model_id}/result'
        )

        # Create inference result record (save both input and output image URLs)
        inference_result = create_inference_result(
            model_id,
            model_name,
            user_email, 
            image_url,  # The URL of the uploaded input image
            annotated_image_url,  # The URL of the annotated image
            json.dumps(labels_json)  # JSON response from YOLO API
            
        )

        if not inference_result:
            return 500, {"message": "Failed to create inference result"}

        return 200, {
            "message": "Inference performed successfully",
            "inference_id": inference_result['result_id'],
            "image_url": image_url,
            "email": user_email,
            "model_id": model_id,
            "annotated_image_url": annotated_image_url,
            "model_name": model_name,
            "json_response": json.dumps(labels_json)
        }

    except requests.exceptions.RequestException as e:
        # Catch network-related errors or invalid requests
        return 500, {"message": f"Request failed: {str(e)}"}

def run_inference_windscreen(model_id, image_file, user_email):
    try:
        """
        Store the image in the S3 bucket, make the inference via API call,
        and create an inference result record. The output annotated image is also stored in the S3 bucket.
        """
        # Read the file content once
        image_content = image_file.read()
        # Upload image to S3
        try:
            # Rewind original stream so it can be used again (if needed)
            image_file.stream.seek(0)
            image_url = upload_to_s3(image_file, image_file.filename, model_id)
        except RuntimeError as e:
            return 500, {"message": str(e)}

        model_name = "Windscreen Detection Model"
        print("Image URL: ", image_url)


        files = {
        'file': (image_file.filename, image_content, image_file.mimetype)
    }
        data = {
            'weights': "WindscreenExp44.pt"
        }
        # print("Files to be sent:", files)
        print("Data to be sent:", data)

        response = requests.post(f"{INDIVIDUAL_YOLOS_API}/infer_crack", files=files)
        if response.status_code != 200:
            return response.status_code, {"message": "Inference failed", "details": response.json()}
        
        data = response.json()
        annotated_image_relative_url = data.get('image_url')
        crack_detected = data.get('crack_detected', False)
        windscreen_detected = data.get('windscreen_detected', False)
        labels_json = data.get('labels', {})
        
        annotated_image_url = None
        if annotated_image_relative_url:
            # Fetch annotated image from YOLO API
            annotated_image_response = requests.get(f'{INDIVIDUAL_YOLOS_API}/{annotated_image_relative_url}')
            if annotated_image_response.status_code != 200:
                return 500, {"message": "Failed to fetch annotated image from YOLO API"}
            
            annotated_image_filename = os.path.basename(annotated_image_relative_url)
            # Upload the fetched image to S3
            annotated_image_url = upload_to_s3(
                BytesIO(annotated_image_response.content),
                annotated_image_filename,
                folder='windscreen/result'
            )
        
        inference_msg = None
        print("windscreen_detected: ", windscreen_detected)
        print("crack_detected: ", crack_detected)


        if windscreen_detected and not crack_detected:
            inference_msg = "Windscreen detected with no cracks"
        elif windscreen_detected and crack_detected:
            inference_msg =json.dumps(labels_json)  
        else:
            inference_msg = "No windscreen detected"
        print("Inference Message: ----------------------------------------------------", inference_msg)
        inference_result = create_inference_result(
            model_id=model_id,
            model_name=model_name,
            user_email=user_email,
            input_image_url=image_url,
            output_image_url=annotated_image_url,
            json_response=inference_msg
        )

        if not inference_result:
            return 500, {"message": "Failed to create inference result"}
        
        return 200, {
            "message": "Inference performed successfully",
            "inference_id": inference_result['result_id'],
            "image_url": image_url,
            "email": user_email,
            "model_id": model_id,
            "annotated_image_url": annotated_image_url,
            "model_name": model_name,
            "windscreen_detected": windscreen_detected,
            "crack_detected": crack_detected,
            "json_response": json.dumps(labels_json)
        }

    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Request failed: {str(e)}"}


def run_inference_stoplight(model_id, image1_file, image2_file, user_email):
    print("Running inference for stoplight detection")
    # try:
    #     """
    #     Store the images in the S3 bucket, make the inference via API call,
    #     and create an inference result record. The output annotated image is also stored in the S3 bucket.
    #     """
    #     # Read the file content once
    #     image1_content = image1_file.read()
    #     image2_content = image2_file.read()

    #     # Upload images to S3
    #     try:
    #         # Rewind original streams so they can be used again (if needed)
    #         image1_file.stream.seek(0)
    #         image2_file.stream.seek(0)
    #         image1_url = upload_to_s3(image1_file, image1_file.filename, model_id)
    #         image2_url = upload_to_s3(image2_file, image2_file.filename, model_id)
    #     except RuntimeError as e:
    #         return 500, {"message": str(e)}

    #     model_name = "Stoplight Detection Model"
    #     print("Image URLs: ", image1_url, image2_url)

    #     files = {
    #         'file1': (image1_file.filename, image1_content, image1_file.mimetype),
    #         'file2': (image2_file.filename, image2_content, image2_file.mimetype)
    #     }
        
    #     data = {
    #         'weights': "StoplightExp44.pt"
    #     }
        
    #     print("Files to be sent:", files)
    #     print("Data to be sent:", data)

    #     response = requests.post(f"{INDIVIDUAL_YOLOS_API}/infer_stoplight", files=files)
    #     if response.status_code != 200:
    #         return response.status_code, {"message": "Inference failed", "details": response.json()}
        
    #     data = response.json()
    #     annotated_image_relative_url = data.get('image_url')
    #     stoplight_detected = data.get('stoplight_detected', False)
    #     labels_json = data.get('labels', {})
        
    #     annotated_image_url = None
    #     if annotated_image_relative_url:
    #         # Fetch annotated image from YOLO API
    #         annotated_image_response = requests.get(f'{INDIVIDUAL_YOLOS_API}/{annotated_image_relative_url}')
    #         if annotated_image_response.status_code != 200:
    #             return 500, {"message": "Failed to fetch annotated image from YOLO API"}
            
    #         annotated_image_filename = os.path.basename(annotated_image_relative_url)