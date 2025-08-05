import os
import base64
import google.generativeai as genai
from openai import OpenAI
import fitz  # PyMuPDF
import json
from PIL import Image
import io
import tempfile
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# model = "gpt-4.1-mini-2025-04-14"  
model = "gpt-4.1-2025-04-14" 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resize_image_for_llm(image, max_size=(900, 900), quality=95):
    """Resize image to fit within max_size box while maintaining aspect ratio"""
    original_size = image.size
    print(f"Original image size: {original_size}")
    
    # Calculate ratios for both dimensions
    width_ratio = max_size[0] / original_size[0]
    height_ratio = max_size[1] / original_size[1]
    
    # Use the smaller ratio to ensure image fits within the box
    ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = int(original_size[0] * ratio)
    new_height = int(original_size[1] * ratio)
    
    # Resize with exact proportions
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized image size: {resized_image.size} (ratio: {ratio:.3f})")
    
    # Convert to RGB if needed
    if resized_image.mode in ("RGBA", "P"):
        rgb_image = Image.new("RGB", resized_image.size, (255, 255, 255))
        if resized_image.mode == "P":
            resized_image = resized_image.convert("RGBA")
        rgb_image.paste(resized_image, mask=resized_image.split()[-1] if resized_image.mode == "RGBA" else None)
        resized_image = rgb_image
    
    # Save resized image to see the size
    # resized_image.save(f"resized_page_{int(time.time())}.jpg", "JPEG", quality=quality)
    
    # Convert to base64
    buffered = io.BytesIO()
    resized_image.save(buffered, format="JPEG", quality=quality, optimize=True)
    file_size_kb = len(buffered.getvalue()) / 1024
    print(f"Compressed image size: {file_size_kb:.1f} KB")
    
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return resized_image, image_base64


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
    # buffered = io.BytesIO()
    # image.save(buffered, format="PNG")
    # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    resized_image, image_base64 = resize_image_for_llm(image)
    
    prompt = """
    Look at this document and determine if it contains invoice information.
    
    An invoice should have:
    1. The word "INVOICE" visible on header/somewhere in the document
    2. A company name providing the service
    3. Billing amounts or charges
    4. An invoice/tracking/PRO number
    
    REJECT only if:
    - This is clearly a delivery receipt (says "DELIVERY RECEIPT" at top)
    - This is a packing slip without billing information
    - This page has no billing/charging information at all
    - This page has no "INVOICE" text anywhere
    
    Strictly follow the above rules, do not make assumptions or guesses for invoice detection.
    If you see "INVOICE" text and billing information, answer "yes".
    If this is clearly not an invoice document, answer "no".
    """
    
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,  # Using available model instead of the requested one
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.1,
            )
            
            # Get response text
            result = response.choices[0].message.content.lower().strip()
            result = result.replace(".", "").replace(",", "").replace("!", "")
            
            print(f"Invoice detection attempt {attempt + 1}: '{result}'")
            
            # Strict yes/no checking
            if result == "yes":
                return True
            elif result == "no":
                return False
            elif "yes" in result and "no" not in result:
                return True
            elif "no" in result and "yes" not in result:
                return False
            else:
                # Ambiguous response, retry
                if attempt < max_retries - 1:
                    print(f"Ambiguous response: '{result}', retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"Final ambiguous response, defaulting to False")
                    return False
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"All attempts failed, defaulting to False")
                return False
            time.sleep(2)
    
    return False

def extract_invoice_data(image, page_num):
    # Convert image to base64 for API request
    # buffered = io.BytesIO()
    # image.save(buffered, format="PNG")
    # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    resized_image, image_base64 = resize_image_for_llm(image)
    
    prompt = """
    You are Expert to analyze this invoice image and extract the following information precisely. 
    
    
    Fields to extract:
    1. **Company Name**: Extract from header/logo area - the company providing the trucking service
   
    2. **Tracking/Invoice Number**: Look for these EXACT labels:
        - "INVOICE #:" or "INVOICE#:"
        - "ORIGINAL INVOICE"  
        - "Invoice Number:" or "Invoice No:"
        - "Tracking Number:" or "Pro#"
        - Extract the alphanumeric code that follows these labels

    3. **Order Number** (Shipper-related): Look for these EXACT labels ONLY:
        - "SHIPPER NUMBER" or "SHIPPER:"
        - "Shipper Number:"
        - "Order Number:" or "Order No:"
        - "Reference Number:" 
        - "ORD" or "ORD#"
        - "B/L" (Bill of Lading)
        - CRITICAL: DO NOT use any field labeled with "PO" for this

    4. **Customer PO Number**: Look for these EXACT labels ONLY:
        - "P.O. NUMBER" or "PO#:"
        - "PO Number:" or "PO:"
        - "Purchase Order:"
        - "Customer PO:"
        - CRITICAL: DO NOT use Shipper numbers for this field
        - must be in the format of a PO number (alphanumeric, only letters, only numbers) not sentences or words

    5. **Total Charges**: Look for these EXACT labels:
        - "REMIT AMT:" 
        - "PLEASE PAY THIS AMOUNT"
        - "Total Due:" or "Total:"
        - "Amount Due:" or "Balance Due:"
        - "SHIPMENT TOTAL:"
        - "TOTAL AMOUNT:"
        - "INVOICE TOTAL:"
        - Extract with dollar sign ($)
    
    VALIDATION: Before extracting, verify this is an actual invoice:
    - check the all labels for the each field mentioned above
    - Must have "INVOICE" in the header of the document
    - Must have billing/charging information
    - Must not be a delivery receipt or packing slip

    CRITICAL DISAMBIGUATION RULES:
    - If you see "PO Number:" or "PO:" → This goes to "Customer Po number" field
    - If you see "Shipper Number:" or "Shipper:" → This goes to "Order number" field  
    - DO NOT put PO numbers in the Order number field
    - DO NOT put Shipper numbers in the Customer Po number field
    - If field appears in multiple locations, use the value from the main invoice area
    
    Critical Rule:
    - Always check for the labels mentioned above before extracting any field
    - Map the correct/exact label to the exact field, do not guess or assume
    - PO number and Order number are different, do not confuse or mix them 
    
    Mapping Rules:
    - for "Customer Po number" --> The customer PO number (point 4)
    - for "Trucking Company name" --> The company name (point 1)
    - for "Order number" --> The order number (point 3)             // do not consider PO number as order number
    - for "Tracking/invoice number" --> The tracking number (point 2)
    - for "Total Charges" --> The total charges amount (point 5)
    
    
    If this IS a valid invoice, return the information in this JSON format:
    {
        "page_no": [PAGE NUMBER],
        "is_valid_invoice": true,
        "Trucking Company name": "[EXACT COMPANY NAME]",
        "Order number": "[SHIPPER/ORDER NUMBER ONLY]",  // do not consider PO number as order number
        "Tracking number": "[INVOICE/TRACKING NUMBER]", 
        "Customer Po number": "[PO NUMBER ONLY]",
        "Total Charges": "[AMOUNT WITH $ SIGN]"
    }
    
    IMPORTANT INSTRUCTIONS: 
    - Follow the as is rule for field extraction as mentioned above
    - for each field, look for the mentioned labels in the document
    - Do NOT include any markdown formatting like ```json or ``` in your response
    - Check each field carefully and ensure it is the label exact match
    - Include the dollar sign ($) with the Total Charges

    
    **Strictly follow the above rules, do not hallucinate on any field extraction or do not repeat the labels in your response.**
    """
    
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model= model, # Using available model instead of the requested one
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1, 
            )
            
            # Get response text
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            print(f"Extraction attempt {attempt + 1}: Response length: {len(response_text)}")
            
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON parse error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e
            
            # Check if this is marked as invalid invoice
            if result.get("is_valid_invoice") == False:
                return {
                    "page_no": page_num,
                    "is_valid_invoice": False,
                    "skip_reason": "Not a valid invoice page",
                    "Trucking Company name": None,
                    "Order number": None,
                    "Tracking number": None,
                    "Customer Po number": None,
                    "Total Charges": None
                }
            
            # Ensure all required fields are present
            required_fields = ["Trucking Company name", "Order number", "Tracking number", "Customer Po number", "Total Charges"]
            for field in required_fields:
                if field not in result:
                    result[field] = None
            
            # Set page number and validation flag
            result["page_no"] = page_num
            result["is_valid_invoice"] = True
            
            # Format Total Charges with dollar sign if not present and not null
            if result["Total Charges"] and not isinstance(result["Total Charges"], str):
                result["Total Charges"] = f"${result['Total Charges']}"
            elif result["Total Charges"] and isinstance(result["Total Charges"], str) and not result["Total Charges"].startswith("$"):
                result["Total Charges"] = f"${result['Total Charges']}"
            
            # Validate that we have meaningful data
            meaningful_fields = [
                result.get("Trucking Company name"),
                result.get("Tracking number"), 
                result.get("Total Charges")
            ]
            
            # If we have at least one meaningful field, return the result
            if any(field and str(field).lower() not in ['null', 'none', ''] for field in meaningful_fields):
                print(f"Extraction successful on attempt {attempt + 1}")
                return result
            else:
                print(f"No meaningful data found on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    # Return the result even if no meaningful data
                    return result
            
        except Exception as e:
            print(f"Extraction attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to extract data after {max_retries} attempts")
                return {
                    "error": "Failed to parse invoice data",
                    "raw_response": response.choices[0].message.content if 'response' in locals() else "No response",
                    "page_no": page_num,
                    "is_valid_invoice": False
                }
            time.sleep(3)
    
    # Fallback return
    return {
        "error": "No successful extraction",
        "page_no": page_num,
        "is_valid_invoice": False
    }

def recheck_null_fields(image, page_num, current_data, null_fields):
    """Re-extract only the null fields from a specific page"""
    
    # buffered = io.BytesIO()
    # image.save(buffered, format="PNG")
    # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    resized_image, image_base64 = resize_image_for_llm(image)

    
    # Create focused prompt for only the null fields
    field_descriptions = {
        "Trucking Company name": "company name from the header/title area of the document (the company providing the service)",
        "Order number": "order/shipper number (look for labels: SHIPPER NUMBER, SHIPPER:, Shipper Number:, Order Number, Order No, Reference Number, ORD, ORD#, B/L) - EXCLUDE any PO labels",
        "Tracking number": "invoice/tracking number (look for labels: INVOICE#, ORIGINAL INVOICE, Invoice Number, Invoice No, Tracking Number, Pro#)",
        "Customer Po number": "PO number ONLY (look for labels: P.O. NUMBER, PO#:, PO Number:, PO:, Purchase Order, Customer PO) - DO NOT use Shipper numbers",
        "Total Charges": "total amount to be paid (look for labels: PLEASE PAY THIS AMOUNT, REMIT AMT:, Total Due, Total, Amount Due, Balance Due, Total Amount)"
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
        response = client.chat.completions.create(
            model= model,  # Using available model instead of the requested one
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        
        response_text = response.choices[0].message.content.strip()
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
    found_primary_invoice = False  # Track if we found the main invoice
    
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"Processing page {page_num}...")
        
        # If we already found a primary invoice, be more strict about additional pages
        if found_primary_invoice:
            print(f"Page {page_num} - Primary invoice already found, checking if this is a valid secondary invoice...")
        
        try:
            # Check if the page is an invoice with better validation
            if is_invoice_page(image):
                print(f"Page {page_num} is identified as an invoice. Extracting data...")
                
                # Extract data from the invoice
                invoice_data = extract_invoice_data(image, page_num)
                
                
                # Check if extraction marked this as invalid
                if invoice_data.get("is_valid_invoice") == False:
                    print(f"Page {page_num} - Marked as invalid invoice during extraction. Skipping...")
                    continue
                
                # Validate that we got meaningful data
                has_meaningful_data = any(
                    v and str(v).lower() not in ['null', 'none', ''] 
                    for k, v in invoice_data.items() 
                    if k in ["Trucking Company name", "Tracking number", "Total Charges"]
                )
                
                if has_meaningful_data:
                    # Check for duplicate invoice detection
                    if found_primary_invoice:
                        # Check if this is a duplicate of already found invoice
                        existing_tracking = [r.get("Tracking number") for r in results if r.get("Tracking number")]
                        current_tracking = invoice_data.get("Tracking number")
                        
                        if current_tracking and current_tracking in existing_tracking:
                            print(f"Page {page_num} - Duplicate invoice detected (same tracking number). Skipping...")
                            continue
                        else:
                            print(f"Page {page_num} - Additional unique invoice found")
                    else:
                        found_primary_invoice = True
                        print(f"Page {page_num} - Primary invoice found")
                    
                    results.append(invoice_data)
                    print(f"Page {page_num} - Successfully extracted data")
                else:
                    print(f"Page {page_num} - No meaningful data extracted, skipping...")
            else:
                print(f"Page {page_num} is not an invoice. Skipping...")
                
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
            continue
    
    # Filter out invalid invoices before processing
    valid_results = [r for r in results if r.get("is_valid_invoice") != False]
    
    if not valid_results:
        print("No valid invoices found in the PDF")
        return []
    
    print(f"Found {len(valid_results)} valid invoice(s) in the PDF")
    
    # Clean up data first
    cleaned_results = validate_and_cleanup_data(valid_results)
    
    # Final null validation - check and fill null values
    final_results = final_null_validation(cleaned_results, images)
    
    # Final deduplication check
    # final_results = remove_duplicate_invoices(final_results)

    return final_results

def validate_and_cleanup_data(invoice_data_list):
    cleaned_data = []
    
    for invoice in invoice_data_list:
        # Skip entries with errors or invalid invoices
        if "error" in invoice or invoice.get("is_valid_invoice") == False:
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

def remove_duplicate_invoices(invoice_data_list):
    """Remove duplicate invoices based on tracking number and total charges"""
    
    if not invoice_data_list:
        return invoice_data_list
    
    unique_invoices = []
    seen_combinations = set()
    
    for invoice in invoice_data_list:
        tracking_num = invoice.get("Tracking number")
        total_charges = invoice.get("Total Charges")
        
        # Create a unique identifier
        identifier = f"{tracking_num}_{total_charges}"
        
        if identifier not in seen_combinations:
            seen_combinations.add(identifier)
            unique_invoices.append(invoice)
            print(f"Keeping invoice from page {invoice.get('page_no')} - Tracking: {tracking_num}")
        else:
            print(f"Removing duplicate invoice from page {invoice.get('page_no')} - Tracking: {tracking_num}")
    
    print(f"Deduplication: {len(invoice_data_list)} -> {len(unique_invoices)} invoices")
    return unique_invoices

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
        
        cost_per_image = 0.00153  # Cost per image in USD
        pages_with_2_calls = 0
        pages_with_1_call = 0
        total_api_calls = 0
        
        # Process each page
        results = []
        
        for i, image in enumerate(image_list):
            page_num = i + 1
            print(f"Processing page {page_num}...")
            
            # ADDED: If we already found a primary invoice, be more strict about additional pages
            if len(results) > 0:
                print(f"Page {page_num} - Primary invoice already found, checking if this is a valid secondary invoice...")
            
            try:
                # ENHANCED: Better invoice validation
                if is_invoice_page(image):
                    print(f"Page {page_num} is identified as an invoice. Extracting data...")
                    
                    invoice_data = extract_invoice_data(image, page_num)
                    
                    pages_with_2_calls += 1
                    total_api_calls += 2
                    
                    # ADDED: Check if extraction marked this as invalid
                    if invoice_data.get("is_valid_invoice") == False:
                        print(f"Page {page_num} - Marked as invalid invoice during extraction. Skipping...")
                        continue
                    
                    # Validate meaningful data
                    has_meaningful_data = any(
                        v and str(v).lower() not in ['null', 'none', ''] 
                        for k, v in invoice_data.items() 
                        if k in ["Trucking Company name", "Tracking number", "Total Charges"]
                    )
                    
                    if has_meaningful_data:
                        # ADDED: Check for duplicate invoice detection
                        if len(results) > 0:
                            # Check if this is a duplicate of already found invoice
                            existing_tracking = [r.get("Tracking number") for r in results if r.get("Tracking number")]
                            current_tracking = invoice_data.get("Tracking number")
                            
                            if current_tracking and current_tracking in existing_tracking:
                                print(f"Page {page_num} - Duplicate invoice detected (same tracking number). Skipping...")
                                continue
                            else:
                                print(f"Page {page_num} - Additional unique invoice found")
                        else:
                            print(f"Page {page_num} - Primary invoice found")
                        
                        results.append(invoice_data)
                        print(f"Page {page_num} - Successfully extracted data")
                    else:
                        print(f"Page {page_num} - No meaningful data extracted, skipping...")
                else:
                    print(f"Page {page_num} is not an invoice. Skipping...")
                    
                    pages_with_1_call += 1
                    total_api_calls += 1
                    
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
        
        # ADDED: Filter out invalid invoices
        valid_results = [r for r in results if r.get("is_valid_invoice") != False]
        
        if not valid_results:
            print("No valid invoices found in the PDF")
            return jsonify([])
        
        # Validate and clean up the data
        cleaned_data = validate_and_cleanup_data(valid_results)
        
        # Final null validation - check and fill null values
        final_data = final_null_validation(cleaned_data, image_list)
        
        # ADDED: Final deduplication check
        # final_data = remove_duplicate_invoices(final_data)
        
        print(f"Found {len(final_data)} valid unique invoice(s) in the PDF")
        
        total_cost = total_api_calls * cost_per_image
        print("\n" + "="*50)
        print("💰 COST CALCULATION SUMMARY")
        print("="*50)
        print(f"📄 Total pages processed: {page_count}")
        print(f"📋 Invoice pages (2 calls): {pages_with_2_calls}")
        print(f"📄 Non-invoice pages (1 call): {pages_with_1_call}")
        print(f"🔄 Total API calls: {total_api_calls}")
        print(f"💵 Cost per image: ${cost_per_image}")
        print(f"💰 TOTAL COST: ${total_cost:.6f}")
        print(f"✅ Successfully extracted: {len(final_data)} invoices")
        print("="*50)
        
        return jsonify(final_data)
        
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)