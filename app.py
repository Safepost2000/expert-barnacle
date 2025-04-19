import os
import io
import re
import json
import base64
import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai
from PIL import Image
import tempfile
import openpyxl
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import datetime
import time
import zipfile

# Load environment variables and configure API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Advanced Invoice Processing",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .agent-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .agent-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .status {
        font-weight: bold;
    }
    .success {
        color: #4CAF50;
    }
    .error {
        color: #F44336;
    }
    .info {
        color: #2196F3;
    }
    .warning {
        color: #FF9800;
    }
    .file-box {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .tab-content {
        padding: 1rem;
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 5px 5px;
    }
    .nav-item {
        cursor: pointer;
        padding: 0.5rem 1rem;
        background-color: #f0f8ff;
        border: 1px solid #ddd;
        border-radius: 5px 5px 0 0;
        margin-right: 5px;
    }
    .nav-item.active {
        background-color: #1E88E5;
        color: white;
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>Advanced Invoice Processing System</div>", unsafe_allow_html=True)
st.markdown("""
This application uses three AI agents to process invoice documents, including handwritten notes:
1. **Invoice Reader**: Extracts data from invoice images/PDFs (including handwritten content)
2. **Excel Feeder**: Maps and enters data into Excel spreadsheets
3. **Data Validator**: Verifies data accuracy and reports any issues
""")

# Initialize session state variables if they don't exist
if 'processed_invoices' not in st.session_state:
    st.session_state.processed_invoices = {}
if 'current_invoice_id' not in st.session_state:
    st.session_state.current_invoice_id = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'tally_xml' not in st.session_state:
    st.session_state.tally_xml = None

# Initialize Gemini models
def initialize_vision_model(model_name="gemini-pro-vision"):
    """Initialize and return the specified Gemini vision model."""
    model = genai.GenerativeModel(model_name)
    return model

def initialize_text_model(model_name="gemini-pro"):
    """Initialize and return the text-only Gemini model."""
    model = genai.GenerativeModel(model_name)
    return model

# AGENT 1: Invoice Reader
class InvoiceReaderAgent:
    def __init__(self):
        self.model = initialize_vision_model("gemini-pro-vision")
        self.name = "Invoice Reader"
    
    def extract_data(self, image_bytes, image_type, file_name=""):
        """Extract invoice information from the uploaded document, including handwritten content."""
        system_prompt = """
        You are an expert invoice data extraction agent with special capability to recognize both printed and handwritten text. 
        Analyze the provided invoice image and extract the following information in JSON format:
        1. Invoice Number
        2. Date
        3. Vendor Name
        4. Vendor Address
        5. Buyer Name
        6. Buyer Address
        7. Line Items (as an array of objects with description, quantity, unit_price, and total)
        8. Subtotal
        9. Tax
        10. Tax Rate (if available)
        11. Discount (if available)
        12. Total Amount
        13. Currency
        14. Payment Method (if available)
        15. Payment Terms (if available)
        16. Due Date (if available)
        17. Handwritten Notes (capture any handwritten annotations on the invoice)
        
        Pay special attention to:
        - Handwritten notes or corrections on printed invoices
        - Fully handwritten invoices or receipts
        - Any annotations or modifications made by hand
        
        Return ONLY a valid JSON object with these fields. If you cannot find a particular field, use null as its value.
        The JSON format should be:
        {
            "invoice_number": "value",
            "date": "value",
            "vendor_name": "value",
            "vendor_address": "value",
            "buyer_name": "value",
            "buyer_address": "value",
            "line_items": [
                {
                    "description": "value",
                    "quantity": number,
                    "unit_price": number,
                    "total": number
                }
            ],
            "subtotal": number,
            "tax": number,
            "tax_rate": "value",
            "discount": number,
            "total": number,
            "currency": "value",
            "payment_method": "value",
            "payment_terms": "value",
            "due_date": "value",
            "handwritten_notes": "value"
        }
        """
        
        if file_name:
            system_prompt += f"\n\nThe file name is: {file_name}"
        
        image_info = [
            {
                "mime_type": image_type,
                "data": image_bytes
            }
        ]
        
        try:
            response = self.model.generate_content([system_prompt, image_info[0]])
            response_text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_text
            
            json_str = json_str.strip()
            extracted_data = json.loads(json_str)
            
            extracted_data["_metadata"] = {
                "extraction_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": file_name,
                "model": "gemini-pro-vision"
            }
            
            return extracted_data
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

# AGENT 2: Excel Feeder
class ExcelFeederAgent:
    def __init__(self):
        self.model = initialize_text_model("gemini-pro")
        self.name = "Excel Feeder"
    
    def map_data_to_excel(self, extracted_data):
        """Map the extracted data to Excel format."""
        try:
            if "error" in extracted_data:
                return {"error": extracted_data["error"]}
            
            main_info = {
                "Field": ["Invoice Number", "Date", "Vendor Name", "Vendor Address", "Buyer Name", "Buyer Address",
                          "Subtotal", "Tax", "Tax Rate", "Discount", "Total Amount", "Currency", "Payment Method",
                          "Payment Terms", "Due Date", "Handwritten Notes"],
                "Value": [
                    extracted_data.get("invoice_number", ""),
                    extracted_data.get("date", ""),
                    extracted_data.get("vendor_name", ""),
                    extracted_data.get("vendor_address", ""),
                    extracted_data.get("buyer_name", ""),
                    extracted_data.get("buyer_address", ""),
                    extracted_data.get("subtotal", ""),
                    extracted_data.get("tax", ""),
                    extracted_data.get("tax_rate", ""),
                    extracted_data.get("discount", ""),
                    extracted_data.get("total", ""),
                    extracted_data.get("currency", ""),
                    extracted_data.get("payment_method", ""),
                    extracted_data.get("payment_terms", ""),
                    extracted_data.get("due_date", ""),
                    extracted_data.get("handwritten_notes", "")
                ]
            }
            main_df = pd.DataFrame(main_info)
            
            line_items = extracted_data.get("line_items", [])
            if line_items:
                items_df = pd.DataFrame(line_items)
            else:
                items_df = pd.DataFrame(columns=["description", "quantity", "unit_price", "total"])
            
            return {
                "main_info": main_df,
                "line_items": items_df
            }
        except Exception as e:
            return {"error": f"Excel mapping failed: {str(e)}"}
    
    def create_excel_file(self, excel_data, invoice_id="invoice"):
        """Create an Excel file with the mapped data."""
        try:
            if "error" in excel_data:
                return {"error": excel_data["error"]}
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                excel_data["main_info"].to_excel(writer, sheet_name="Invoice Information", index=False)
                excel_data["line_items"].to_excel(writer, sheet_name="Line Items", index=False)
                
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for i, col in enumerate(excel_data["main_info" if sheet_name == "Invoice Information" else "line_items"].columns):
                        column_width = max(excel_data["main_info" if sheet_name == "Invoice Information" else "line_items"][col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.column_dimensions[openpyxl.utils.get_column_letter(i+1)].width = column_width
            
            output.seek(0)
            return output
        except Exception as e:
            return {"error": f"Excel creation failed: {str(e)}"}
    
    def create_tally_import_format(self, extracted_data):
        """Create Tally Prime import format (XML) from the extracted data."""
        try:
            if "error" in extracted_data:
                return {"error": extracted_data["error"]}
            
            invoice_date = extracted_data.get("date", "")
            try:
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%B %d, %Y"]:
                    try:
                        date_obj = datetime.datetime.strptime(invoice_date, fmt)
                        formatted_date = date_obj.strftime("%Y%m%d")
                        break
                    except ValueError:
                        continue
                else:
                    formatted_date = datetime.datetime.now().strftime("%Y%m%d")
            except:
                formatted_date = datetime.datetime.now().strftime("%Y%m%d")
            
            tally_xml = f"""
<ENVELOPE>
  <HEADER>
    <TALLYREQUEST>Import Data</TALLYREQUEST>
  </HEADER>
  <BODY>
    <IMPORTDATA>
      <REQUESTDESC>
        <REPORTNAME>Vouchers</REPORTNAME>
        <STATICVARIABLES>
          <SVCURRENTCOMPANY>My Company</SVCURRENTCOMPANY>
        </STATICVARIABLES>
      </REQUESTDESC>
      <REQUESTDATA>
        <TALLYMESSAGE>
          <VOUCHER VCHTYPE="Sales" ACTION="Create">
            <DATE>{formatted_date}</DATE>
            <NARRATION>Invoice: {extracted_data.get("invoice_number", "")}</NARRATION>
            <VOUCHERTYPENAME>Sales</VOUCHERTYPENAME>
            <VOUCHERNUMBER>{extracted_data.get("invoice_number", "")}</VOUCHERNUMBER>
            <REFERENCE>{extracted_data.get("invoice_number", "")}</REFERENCE>
            <PARTYLEDGERNAME>{extracted_data.get("vendor_name", "")}</PARTYLEDGERNAME>
            <EFFECTIVEDATE>{formatted_date}</EFFECTIVEDATE>
            <ALLLEDGERENTRIES.LIST>
              <LEDGERNAME>Sales</LEDGERNAME>
              <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
              <AMOUNT>-{extracted_data.get("subtotal", 0)}</AMOUNT>
            </ALLLEDGERENTRIES.LIST>
            <ALLLEDGERENTRIES.LIST>
              <LEDGERNAME>Tax</LEDGERNAME>
              <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
              <AMOUNT>-{extracted_data.get("tax", 0)}</AMOUNT>
            </ALLLEDGERENTRIES.LIST>
            <ALLLEDGERENTRIES.LIST>
              <LEDGERNAME>{extracted_data.get("vendor_name", "Customer")}</LEDGERNAME>
              <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
              <AMOUNT>{extracted_data.get("total", 0)}</AMOUNT>
            </ALLLEDGERENTRIES.LIST>
            """
            
            line_items = extracted_data.get("line_items", [])
            for item in line_items:
                tally_xml += f"""
            <INVENTORYENTRIES.LIST>
              <STOCKITEMNAME>{item.get("description", "Item")}</STOCKITEMNAME>
              <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
              <RATE>{item.get("unit_price", 0)}</RATE>
              <AMOUNT>-{item.get("total", 0)}</AMOUNT>
              <ACTUALQTY>{item.get("quantity", 0)}</ACTUALQTY>
              <BILLEDQTY>{item.get("quantity", 0)}</BILLEDQTY>
              <BATCHALLOCATIONS.LIST>
                <GODOWNNAME>Main Location</GODOWNNAME>
                <BATCHNAME>Primary Batch</BATCHNAME>
                <AMOUNT>-{item.get("total", 0)}</AMOUNT>
                <ACTUALQTY>{item.get("quantity", 0)}</ACTUALQTY>
                <BILLEDQTY>{item.get("quantity", 0)}</BILLEDQTY>
              </BATCHALLOCATIONS.LIST>
            </INVENTORYENTRIES.LIST>
                """
            
            tally_xml += """
          </VOUCHER>
        </TALLYMESSAGE>
      </REQUESTDATA>
    </IMPORTDATA>
  </BODY>
</ENVELOPE>
            """
            
            return tally_xml
        except Exception as e:
            return {"error": f"Tally format creation failed: {str(e)}"}

# AGENT 3: Data Validator
class DataValidatorAgent:
    def __init__(self):
        self.model = initialize_text_model("gemini-pro")
        self.name = "Data Validator"
    
    def validate_data(self, extracted_data, excel_data):
        """Validate the extracted data against the original invoice."""
        if "error" in extracted_data or "error" in excel_data:
            error_message = extracted_data.get("error", "") or excel_data.get("error", "")
            return {"status": "error", "message": error_message, "incidents": [error_message]}
        
        system_prompt = """
        You are a data validation expert with specialization in invoice processing. Analyze the extracted invoice data and identify any potential issues or inconsistencies.
        
        Check the following:
        1. Are all required fields present? (Invoice Number, Date, Vendor Name, Buyer Name, Line Items, Subtotal, Tax, Total Amount)
        2. Are the numerical calculations correct? (Sum of line items should equal subtotal, subtotal + tax - discount should equal total)
        3. Are there any suspicious or unusual values?
        4. Are the handwritten notes properly captured and do they affect the invoice validity?
        5. Is the date format consistent and valid?
        6. Is the invoice number in an expected format?
        7. Is the currency a valid code?
        8. Are the addresses properly formatted?
        9. Is the tax rate consistent with the tax amount?
        10. Is the discount applied correctly?
        
        Return your analysis in JSON format with the following structure:
        {
            "status": "success" or "error",
            "message": "Your overall assessment",
            "incidents": [
                "Description of issue 1",
                "Description of issue 2"
            ],
            "data_quality_score": number (0-100),
            "confidence": number (0-100),
            "handwriting_assessment": "Assessment of any handwritten elements"
        }
        
        If no issues are found, return:
        {
            "status": "success",
            "message": "Invoice processed with no errors.",
            "incidents": [],
            "data_quality_score": 100,
            "confidence": 95,
            "handwriting_assessment": "No handwritten elements detected" or your assessment
        }
        """
        
        try:
            data_str = json.dumps(extracted_data, indent=2)
            response = self.model.generate_content([
                system_prompt,
                f"Extracted invoice data to validate:\n{data_str}"
            ])
            
            response_text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_text
            
            validation_result = json.loads(json_str.strip())
            return validation_result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Validation failed: {str(e)}",
                "incidents": [f"Validation failed: {str(e)}"],
                "data_quality_score": 0,
                "confidence": 0,
                "handwriting_assessment": "Could not assess handwriting due to validation error"
            }

# Function to process a single invoice
def process_invoice(uploaded_file, invoice_id=None):
    """Process a single uploaded invoice using the three-agent system."""
    if uploaded_file is None:
        return {"error": "No file uploaded"}
    
    if invoice_id is None:
        invoice_id = f"invoice_{int(time.time())}"
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            image_bytes = uploaded_file.getvalue()
            image_type = uploaded_file.type
        elif uploaded_file.type == 'application/pdf':
            # Placeholder for PDF handling (e.g., convert to image)
            image_bytes = uploaded_file.getvalue()
            image_type = 'application/pdf'
        else:
            return {"error": "Unsupported file type"}
        
        reader = InvoiceReaderAgent()
        feeder = ExcelFeederAgent()
        validator = DataValidatorAgent()
        
        progress_placeholder.text("Extracting data from invoice...")
        extracted_data = reader.extract_data(image_bytes, image_type, uploaded_file.name)
        if "error" in extracted_data:
            return extracted_data
        
        progress_placeholder.text("Mapping data to Excel format...")
        excel_data = feeder.map_data_to_excel(extracted_data)
        if "error" in excel_data:
            return excel_data
        
        progress_placeholder.text("Creating Excel file...")
        excel_file = feeder.create_excel_file(excel_data, invoice_id)
        if "error" in excel_file:
            return excel_file
        
        progress_placeholder.text("Creating Tally import format...")
        tally_xml = feeder.create_tally_import_format(extracted_data)
        if "error" in tally_xml:
            return tally_xml
        
        progress_placeholder.text("Validating extracted data...")
        validation_result = validator.validate_data(extracted_data, excel_data)
        
        st.session_state.processed_invoices[invoice_id] = {
            "extracted_data": extracted_data,
            "excel_file": excel_file,
            "tally_xml": tally_xml,
            "validation_result": validation_result
        }
        
        progress_placeholder.text("Processing complete.")
        return {"status": "success", "invoice_id": invoice_id}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Streamlit UI
st.sidebar.title("Upload Invoices")
uploaded_files = st.sidebar.file_uploader("Upload invoice images or PDFs", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        result = process_invoice(uploaded_file)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Processed invoice: {result['invoice_id']}")
            st.download_button(
                label="Download Excel",
                data=st.session_state.processed_invoices[result['invoice_id']]['excel_file'],
                file_name=f"{result['invoice_id']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.download_button(
                label="Download Tally XML",
                data=st.session_state.processed_invoices[result['invoice_id']]['tally_xml'],
                file_name=f"{result['invoice_id']}.xml",
                mime="application/xml"
            )
