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
    
    Fileds to extract:
    1. The company name (look at the header/title area for the company name providing the service)
    
    2. The tracking number / invoice number (look for any of these labels): 
       ["INVOICE#:","ORIGINAL INVOICE", "Invoice Number:", "Invoice No:", "Tracking Number:", "Pro#", etc.]
    
    3. The order number (look for any of these labels ): 
       ["SHIPPER NUMBER", "SHIPPER", "Order Number:", "Order No:", "Reference Number:", "ORD", "ORD#", "BILL TO", etc.] 
    
    4. The customer PO number (look for any of these labels ): 
       ["P.O. NUMBER", "PO#:", "PO Number:", "Purchase Order:", "Customer PO:", etc.]
    
    5. The total charges amount (look for any of these labels ): 
       ["PLEASE PAY THIS AMOUNT", "Total Due:", "Total:", "Amount Due:", "Balance Due:", "Total Amount:", etc.]
    
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
    - Follow the as is rule for field extraction as mentioned above
    - for each field, look for the mentioned labels in the document
    - Do NOT include any markdown formatting like ```json or ``` in your response
    - Check each field carefully and ensure it is the most relevant/exact match
    - Include the dollar sign ($) with the Total Charges
    - If you see multiple potential matches for a field, use the lable mentioned above or the most relevant one
    """
    
    # Use retry mechanism for API calls
    max_retries = 10
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
            time.sleep(4)  # Wait before retrying


def recheck_null_fields(image, page_num, current_data, null_fields):
    """Re-extract only the null fields from a specific page"""
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    generation_config = {"temperature": 0.05, "max_output_tokens": 1000}
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    
    # Create focused prompt for only the null fields
    field_descriptions = {
        "Company name": "Extracted from the header/title area of the document.",
        
        "Order number": "order/shipper number (labels: SHIPPER NUMBER, SHIPPER, Order Number:, Order No:, Reference Number:, ORD, ORD#, BILL TO)",
        
        "Tracking number": "invoice/tracking number (labels: INVOICE#:, ORIGINAL INVOICE, Invoice Number:, Invoice No:, Tracking Number:, Pro#)",
        
        "Customer Po number": "PO number (labels: P.O. NUMBER, PO#:, PO Number:, Purchase Order:, Customer PO:)",
        "Total Charges": "total amount (labels: PLEASE PAY THIS AMOUNT, Total Due:, Total:, Amount Due:, Balance Due:, Total Amount:)"
}

    
    missing_fields = []
    for field in null_fields:
        if field in field_descriptions:
            missing_fields.append(f"- {field}: {field_descriptions[field]}")
    
    prompt = f"""
    Look at this document and find ONLY these missing fields:
    
    {chr(10).join(missing_fields)}
    
    Search every part of the document carefully. Look for any text that matches these field types.
    
    Return ONLY this JSON format:
    {{
      {', '.join([f'"{field}": "value or null"' for field in null_fields])}
    }}
    
    No markdown formatting. If you can't find a field, use null.
    """
    
    try:
        chat = model.start_chat()
        response = chat.send_message([
            prompt,
            {
                "mime_type": "image/png",
                "data": image_base64
            }
        ])
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        print(f"Page {page_num} - Rechecked fields: {result}")
        return result
        
    except Exception as e:
        print(f"Page {page_num} - Recheck failed: {str(e)}")
        return {field: None for field in null_fields}




def process_pdf_invoices(pdf_path):
    # Convert PDF to images
    page_count, images = pdf_to_images(pdf_path)
    print(f"PDF has {page_count} pages")
    
    # Process each page
    results = []
    
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"Processing page {page_num}...")
        
        try:
            # Check if the page is an invoice
            if is_invoice_page(image):
                print(f"Page {page_num} is an invoice. Extracting data...")
                
                # Extract data from the invoice
                invoice_data = extract_invoice_data(image, page_num)
                
                # Validate that we got meaningful data
                has_data = any(v for k, v in invoice_data.items() 
                             if k not in ["page_no", "error"] and v is not None)
                
                if has_data:
                    results.append(invoice_data)
                    print(f"Page {page_num} - Successfully extracted data")
                else:
                    print(f"Page {page_num} - No meaningful data extracted, retrying...")
                    invoice_data = extract_invoice_data(image, page_num)
                    results.append(invoice_data)
            else:
                print(f"Page {page_num} is not an invoice. Skipping...")
                
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
            results.append({
                "page_no": page_num,
                "error": str(e),
                "Trucking Company name": None,
                "Order number": None,
                "Tracking number": None,
                "Customer Po number": None,
                "Total Charges": None
            })
    
    # Clean up data first
    cleaned_results = validate_and_cleanup_data(results)
    
    # *** NEW: Final null validation - check and fill null values ***
    final_results = final_null_validation(cleaned_results, images)
    
    return final_results

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


def final_null_validation(invoice_data_list, images):
    """Check final JSON for null values and re-extract them"""
    
    print("=== FINAL NULL CHECK STARTED ===")
    
    for invoice in invoice_data_list:
        page_num = invoice.get("page_no")
        if not page_num or page_num > len(images):
            continue
            
        # Find which fields are null
        null_fields = []
        for field in ["Trucking Company name", "Order number", "Tracking number", "Customer Po number", "Total Charges"]:
            if invoice.get(field) is None:
                null_fields.append(field)
        
        # If there are null fields, go back to that page and re-check
        if null_fields:
            print(f"Page {page_num} has NULL fields: {null_fields}")
            print(f"Going back to page {page_num} to re-extract...")
            
            try:
                # Get the image for this page
                image = images[page_num - 1]  # Convert to 0-based index
                
                # Re-extract only the null fields
                rechecked_data = recheck_null_fields(image, page_num, invoice, null_fields)
                
                # Update the invoice with any found values
                updated_count = 0
                for field, value in rechecked_data.items():
                    if value is not None and str(value).lower() != "null" and str(value).strip() != "":
                        # Format Total Charges with $
                        if field == "Total Charges" and not str(value).startswith("$"):
                            value = f"${value}"
                        
                        invoice[field] = value
                        print(f"✓ Updated {field}: {value}")
                        updated_count += 1
                
                print(f"Page {page_num}: Updated {updated_count} out of {len(null_fields)} null fields")
                
            except Exception as e:
                print(f"Error rechecking page {page_num}: {str(e)}")
        else:
            print(f"Page {page_num}: No null fields found")
    
    print("=== FINAL NULL CHECK COMPLETED ===")
    return invoice_data_list


def process_pdf(pdf_path, output_path=None):
    # Process the PDF
    invoice_data = process_pdf_invoices(pdf_path)
    
    # Validate and clean up the data
    cleaned_data = validate_and_cleanup_data(invoice_data)
        
    return cleaned_data

@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        file_data = file.read()
        pdf_document = fitz.open("pdf", file_data)
        
        # Convert to images
        page_count = pdf_document.page_count
        image_list = []
        
        for page_number in range(page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_list.append(img)
            print(f"Processed page {page_number + 1} of {page_count}")
        
        pdf_document.close()
        print(f"PDF has {page_count} pages")
        
        # Process each page
        results = []
        
        for i, image in enumerate(image_list):
            page_num = i + 1
            print(f"Processing page {page_num}...")
            
            try:
                if is_invoice_page(image):
                    print(f"Page {page_num} is an invoice. Extracting data...")
                    
                    invoice_data = extract_invoice_data(image, page_num)
                    
                    has_data = any(v for k, v in invoice_data.items() 
                                 if k not in ["page_no", "error"] and v is not None)
                    
                    if has_data:
                        results.append(invoice_data)
                        print(f"Page {page_num} - Successfully extracted data")
                    else:
                        print(f"Page {page_num} - No meaningful data extracted, retrying...")
                        invoice_data = extract_invoice_data(image, page_num)
                        results.append(invoice_data)
                else:
                    print(f"Page {page_num} is not an invoice. Skipping...")
                    
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                results.append({
                    "page_no": page_num,
                    "error": str(e),
                    "Trucking Company name": None,
                    "Order number": None,
                    "Tracking number": None,
                    "Customer Po number": None,
                    "Total Charges": None
                })
        
        # Validate and clean up the data
        cleaned_data = validate_and_cleanup_data(results)
        
        # *** NEW: Final null validation - check and fill null values ***
        final_data = final_null_validation(cleaned_data, image_list)
        
        print(f"Found {len(final_data)} invoices in the PDF")
        
        return jsonify(final_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
