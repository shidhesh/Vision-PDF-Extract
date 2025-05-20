# Invoice Data Extractor

A Python application that extracts key data from invoice PDFs using Google's Generative AI.

## Overview

This application processes PDF invoices, identifies which pages contain actual invoices (vs. other document types), and extracts key information from each invoice page. It uses Google's Gemini model to analyze the invoice images and extract structured data.

The application runs as a Flask web service that accepts PDF uploads via HTTP POST requests and returns extracted invoice data in JSON format.

## Features

- **PDF Processing**: Converts PDF pages to high-quality images for analysis
- **Invoice Detection**: Identifies which pages in a PDF are actual invoices
- **Data Extraction**: Extracts the following fields from each invoice:
  - Trucking Company name
  - Order number
  - Tracking number
  - Customer PO number
  - Total Charges
- **Robust Error Handling**: Built-in retry mechanisms and error recovery
- **RESTful API**: Simple HTTP endpoint for invoice processing

## Prerequisites

- Python 3.6+
- Google Generative AI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/invoice-data-extractor.git
   cd invoice-data-extractor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key as an environment variable:
   ```
   export Google_API_key="your_api_key_here"  # On Windows: set Google_API_key=your_api_key_here
   ```

## Usage

### Running the Web Service

Start the Flask application:

```
python invoice.py
```

The server will start on port 5000 by default.

### Processing Invoices via API

Send a POST request to the `/process-invoice` endpoint with a PDF file:

```
curl -X POST -F "file=@path/to/your/invoice.pdf" http://localhost:5000/process-invoice
```

### Response Format

The API returns a JSON array with extracted data for each invoice page:

```json
[
  {
    "page_no": 1,
    "Trucking Company name": "ACME Logistics",
    "Order number": "ORD-12345",
    "Tracking number": "INV-67890",
    "Customer Po number": "PO-54321",
    "Total Charges": "$250.00"
  },
  {
    "page_no": 3,
    "Trucking Company name": "ACME Logistics",
    "Order number": "ORD-78901",
    "Tracking number": "INV-23456",
    "Customer Po number": "PO-87654",
    "Total Charges": "$175.50"
  }
]
```

## How It Works

1. The PDF is converted to high-resolution images (300 DPI)
2. Each page is analyzed to determine if it's an invoice
3. For invoice pages, the Gemini model extracts key data fields
4. Results are validated, cleaned, and returned in JSON format

## API Endpoint

### POST /process-invoice

**Request:**
- Content-Type: multipart/form-data
- Body: form field "file" containing the PDF to process

**Response:**
- Content-Type: application/json
- Body: Array of extracted invoice data objects

**Example Response:**
```json
[
  {
    "page_no": 1,
    "Trucking Company name": "ACME Logistics",
    "Order number": "ORD-12345",
    "Tracking number": "INV-67890",
    "Customer Po number": "PO-54321",
    "Total Charges": "$250.00"
  }
]
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
flask==2.0.1
pillow==9.0.0
pymupdf==1.19.0
google-generativeai==0.3.0
```

## Configuration

The application uses the following environment variables:

- `Google_API_key`: Your Google Generative AI API key

## Limitations

- The model works best with clearly structured invoices
- Processing time depends on the number of pages and complexity
- Requires a valid Google Generative AI API key with sufficient quota

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
