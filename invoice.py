import os
import base64
import google.generativeai as genai
import fitz  # PyMuPDF
import json
from PIL import Image
import io
import tempfile
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure the Google Generative AI API
genai.configure(api_key= os.getenv("Google_API_key"))  # Replace with your actual API key

def pdf_to_images(pdf_path):

    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    page_count = pdf_document.page_count
    image_list = []
    
    # Iterate through each page
    for page_number in range(page_count):
        # Get the page
        page = pdf_document[page_number]
        
        # Convert to a high-quality pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_list.append(img)
        
        print(f"Processed page {page_number + 1} of {page_count}")
    
    # Close the PDF
    pdf_document.close()
    
    return page_count, image_list

def is_invoice_page(image):

    # Convert image to base64 for API request
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Configure the Gemini model
    generation_config = {"temperature": 0.2, "max_output_tokens": 100}
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    
    # Prompt to determine if this is an invoice
    prompt = """
    Analyze this document and determine if it is specifically an INVOICE (not a delivery bill, packing slip, or other document).
    
    Look for these invoice indicators:
    1. The presence of "INVOICE" in the header/title
    2. An invoice number or tracking number
    
    If this is clearly an invoice, answer "yes".
    If this is any other type of document (delivery bill, packing slip, order confirmation, etc.), answer "no".
    
    Answer ONLY "yes" or "no" with no additional text.
    """
    
    # Use retry mechanism for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Start a chat and send the image for processing
            chat = model.start_chat()
            response = chat.send_message([
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            ])
            
            # Check if response indicates this is an invoice
            result = response.text.lower().strip()
            return 'yes' in result
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to determine document type after {max_retries} attempts: {str(e)}")
                return False  # Default to False if we can't determine
            print(f"Attempt {attempt+1} failed, retrying: {str(e)}")
            time.sleep(2)  # Wait before retrying

def extract_invoice_data(image, page_num):

    # Convert image to base64 for API request
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Configure the Gemini model
    generation_config = {"temperature": 0.2, "max_output_tokens": 2000}
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    
    # Detailed prompt for invoice data extraction with flexibility for field names
    prompt = """
    Analyze this invoice image and extract the following information precisely. Be flexible about how the information is labeled in the document.
    
    1. The company name (look at the header/title area for the company name providing the service)
    
    2. The tracking number / invoice number (look for any of these labels: 
       "INVOICE#:","ORIGINAL INVOICE", "Invoice Number:", "Invoice No:", "Tracking Number:", "Pro#", etc.)
    
    3. The order number (look for any of these labels: 
       "SHIPPER NUMBER", "SHIPPER", "Order Number:", "Order No:", "Reference Number:", "ORD", "ORD#", "BILL TO", etc.)
    
    4. The customer PO number (look for any of these labels: 
       "P.O. NUMBER", "PO#:", "PO Number:", "Purchase Order:", "Customer PO:", etc.)
    
    5. The total charges amount (look for any of these labels: 
       "PLEASE PAY THIS AMOUNT", "Total Due:", "Total:", "Amount Due:", 
       "Balance Due:", "Total Amount:", etc.)
    
    Return the information in the following JSON format (and ONLY this format with no additional text):
    {
      "page_no": [PAGE NUMBER],
      "Trucking Company name": [COMPANY NAME],
      "Order number": [ORDER NUMBER],
      "Tracking number": [INVOICE/TRACKING NUMBER],
      "Customer Po number": [PO NUMBER],
      "Total Charges": [TOTAL AMOUNT WITH DOLLAR SIGN]
    }
    
    IMPORTANT INSTRUCTIONS: 
    - Do NOT include any markdown formatting like ```json or ``` in your response
    - If any information is not found, use null for that field
    - Include the dollar sign ($) with the Total Charges
    - The goal is to extract the semantic meaning of each field, not just match exact labels
    - If you see multiple potential matches for a field, use the most appropriate one
    - Use your best judgment to identify the correct information
    """
    
    # Use retry mechanism for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Start a chat and send the image for processing
            chat = model.start_chat()
            response = chat.send_message([
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            ])
            
            # Clean response text to remove any markdown or code block formatting
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the cleaned response as JSON
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ["Trucking Company name", "Order number", "Tracking number", "Customer Po number", "Total Charges"]
            for field in required_fields:
                if field not in result:
                    result[field] = None
            
            # Set page number
            result["page_no"] = page_num
            
            # Format Total Charges with dollar sign if not present and not null
            if result["Total Charges"] and not isinstance(result["Total Charges"], str):
                result["Total Charges"] = f"${result['Total Charges']}"
            elif result["Total Charges"] and isinstance(result["Total Charges"], str) and not result["Total Charges"].startswith("$"):
                result["Total Charges"] = f"${result['Total Charges']}"
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to extract data after {max_retries} attempts: {str(e)}")
                return {
                    "error": "Failed to parse invoice data",
                    "raw_response": response.text if 'response' in locals() else "No response",
                    "page_no": page_num
                }
            print(f"Attempt {attempt+1} failed, retrying: {str(e)}")
            time.sleep(2)  # Wait before retrying

def process_pdf_invoices(pdf_path):

    # Convert PDF to images
    page_count, images = pdf_to_images(pdf_path)
    print(f"PDF has {page_count} pages")
    
    # Process each page
    results = []
    
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"Processing page {page_num}...")
        
        # Check if the page is an invoice
        if is_invoice_page(image):
            print(f"Page {page_num} is an invoice. Extracting data...")
            
            # Extract data from the invoice
            invoice_data = extract_invoice_data(image, page_num)
            results.append(invoice_data)
        else:
            print(f"Page {page_num} is not an invoice. Skipping...")
    
    return results

def validate_and_cleanup_data(invoice_data_list):

    cleaned_data = []
    
    for invoice in invoice_data_list:
        # Skip entries with errors
        if "error" in invoice:
            # Try to salvage some information if available
            if "raw_response" in invoice:
                try:
                    # Try to extract JSON from the raw response
                    raw_text = invoice["raw_response"]
                    if isinstance(raw_text, str):
                        # Clean up the text
                        for prefix in ["```json", "```"]:
                            if raw_text.startswith(prefix):
                                raw_text = raw_text[len(prefix):]
                        for suffix in ["```"]:
                            if raw_text.endswith(suffix):
                                raw_text = raw_text[:-len(suffix)]
                        raw_text = raw_text.strip()
                        
                        # Try to parse as JSON
                        cleaned_invoice = json.loads(raw_text)
                        cleaned_invoice["page_no"] = invoice["page_no"]
                        cleaned_data.append(cleaned_invoice)
                        continue
                except:
                    pass
            
            # If we couldn't salvage data, create a minimal entry
            cleaned_data.append({
                "page_no": invoice["page_no"],
                "Trucking Company name": None,
                "Order number": None,
                "Tracking number": None,
                "Customer Po number": None,
                "Total Charges": None
            })
            continue
            
        # Ensure all fields exist
        required_fields = ["page_no", "Trucking Company name", "Order number", "Tracking number", "Customer Po number", "Total Charges"]
        for field in required_fields:
            if field not in invoice or invoice[field] is None:
                invoice[field] = None
                
        # Format Total Charges with dollar sign if missing
        if invoice["Total Charges"] and not isinstance(invoice["Total Charges"], str):
            invoice["Total Charges"] = f"${invoice['Total Charges']}"
        elif invoice["Total Charges"] and not invoice["Total Charges"].startswith("$"):
            invoice["Total Charges"] = f"${invoice['Total Charges']}"
            
        cleaned_data.append(invoice)
    
    return cleaned_data

def process_pdf(pdf_path, output_path=None):
    # Process the PDF
    invoice_data = process_pdf_invoices(pdf_path)
    
    # Validate and clean up the data
    cleaned_data = validate_and_cleanup_data(invoice_data)
        
    return cleaned_data

@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Process the PDF directly from the uploaded file object
        # Create a PyMuPDF document directly from the file data
        file_data = file.read()
        pdf_document = fitz.open("pdf", file_data)
        
        # Convert to images
        page_count = pdf_document.page_count
        image_list = []
        
        # Iterate through each page
        for page_number in range(page_count):
            # Get the page
            page = pdf_document[page_number]
            
            # Convert to a high-quality pixmap
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_list.append(img)
            
            print(f"Processed page {page_number + 1} of {page_count}")
        
        # Close the PDF
        pdf_document.close()
        
        print(f"PDF has {page_count} pages")
        
        # Process each page
        results = []
        
        for i, image in enumerate(image_list):
            page_num = i + 1
            print(f"Processing page {page_num}...")
            
            # Check if the page is an invoice
            if is_invoice_page(image):
                print(f"Page {page_num} is an invoice. Extracting data...")
                
                # Extract data from the invoice
                invoice_data = extract_invoice_data(image, page_num)
                results.append(invoice_data)
            else:
                print(f"Page {page_num} is not an invoice. Skipping...")
        
        # Validate and clean up the data
        cleaned_data = validate_and_cleanup_data(results)
        
        print(f"Found {len(cleaned_data)} invoices in the PDF")
        
        # Return the results
        return jsonify(cleaned_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)