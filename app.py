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
        4. Line Items (as an array of objects with description, quantity, unit_price, and total)
        5. Subtotal
        6. Tax
        7. Total Amount
        8. Payment Terms (if available)
        9. Due Date (if available)
        10. Handwritten Notes (capture any handwritten annotations on the invoice)
        
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
            "total": number,
            "payment_terms": "value",
            "due_date": "value",
            "handwritten_notes": "value"
        }
        """
        
        # Add the filename to help with context
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
            
            # Extract only the JSON object from the response
            response_text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to find JSON directly
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_text
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Parse the JSON string
            extracted_data = json.loads(json_str)
            
            # Add metadata
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
            
            # Create a DataFrame for the main invoice information
            main_info = {
                "Field": ["Invoice Number", "Date", "Vendor Name", "Subtotal", "Tax", "Total Amount", 
                         "Payment Terms", "Due Date", "Handwritten Notes"],
                "Value": [
                    extracted_data.get("invoice_number", ""),
                    extracted_data.get("date", ""),
                    extracted_data.get("vendor_name", ""),
                    extracted_data.get("subtotal", ""),
                    extracted_data.get("tax", ""),
                    extracted_data.get("total", ""),
                    extracted_data.get("payment_terms", ""),
                    extracted_data.get("due_date", ""),
                    extracted_data.get("handwritten_notes", "")
                ]
            }
            main_df = pd.DataFrame(main_info)
            
            # Create a DataFrame for line items
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
            
            # Create a new Excel workbook
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Write the main invoice information to the first sheet
                excel_data["main_info"].to_excel(writer, sheet_name="Invoice Information", index=False)
                
                # Write the line items to the second sheet
                excel_data["line_items"].to_excel(writer, sheet_name="Line Items", index=False)
                
                # Auto-adjust column widths
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
            
            # Create the Tally XML structure
            envelope = ET.Element("ENVELOPE")
            
            # Add header details
            header = ET.SubElement(envelope, "HEADER")
            ET.SubElement(header, "TALLYREQUEST").text = "Import Data"
            
            # Add body
            body = ET.SubElement(envelope, "BODY")
            import_data = ET.SubElement(body, "IMPORTDATA")
            request_desc = ET.SubElement(import_data, "REQUESTDESC")
            ET.SubElement(request_desc, "REPORTNAME").text = "Vouchers"
            ET.SubElement(request_desc, "STATICVARIABLES")
            
            # Add request data
            request_data = ET.SubElement(import_data, "REQUESTDATA")
            
            # Create a sales voucher entry
            voucher = ET.SubElement(request_data, "TALLYMESSAGE")
            ET.SubElement(voucher, "VOUCHER", {"REMOTEID": "", "VCHTYPE": "Sales", "ACTION": "Create"})
            
            # Format as nicely indented XML
            rough_string = ET.tostring(envelope, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            xml_string = reparsed.toprettyxml(indent="  ")
            
            # Prepare the voucher details
            invoice_date = extracted_data.get("date", "")
            try:
                # Try to parse the date in various formats
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
            
            # Prepare the XML with detailed voucher information
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
            
            # Add inventory entries for each line item
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
            
            # Close the XML structure
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
        1. Are all required fields present? (Invoice Number, Date, Vendor Name, Line Items, Subtotal, Tax, Total Amount)
        2. Are the numerical calculations correct? (Sum of line items should equal subtotal, subtotal + tax should equal total)
        3. Are there any suspicious or unusual values?
        4. Are the handwritten notes properly captured and do they affect the invoice validity?
        5. Is the date format consistent and valid?
        6. Is the invoice number in an expected format?
        
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
            # Convert the extracted data to a string for the model
            data_str = json.dumps(extracted_data, indent=2)
            
            # Generate the validation
            response = self.model.generate_content([
                system_prompt,
                f"Extracted invoice data to validate:\n{data_str}"
            ])
            
            # Extract the JSON from the response
            response_text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to find JSON directly
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_text
            
            # Clean up and parse the JSON
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
        # Get file content
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            image_bytes = uploaded_file.getvalue()
            image_type = uploaded_file.type
        elif uploaded_file.type == 'application/pdf':
            # For PDF, we just pass the bytes directly to Gemini
            image_bytes = uploaded_file.getvalue()
            image_type = uploaded_file.type
        else:
            return {"error": "Unsupported file type. Please upload an image or PDF."}
        
        # Initialize results dictionary
        result = {
            "invoice_id": invoice_id,
            "file_name": uploaded_file.name,
            "file_type": uploaded_file.type
        }
        
        # Agent 1: Invoice Reader
        progress_placeholder.progress(0.33)
        status_placeholder.info(f"Agent 1: Invoice Reader is extracting data from '{uploaded_file.name}'...")
        
        invoice_reader = InvoiceReaderAgent()
        extracted_data = invoice_reader.extract_data(image_bytes, image_type, uploaded_file.name)
        result["extracted_data"] = extracted_data
        
        if "error" in extracted_data:
            result["status"] = "error"
            result["message"] = extracted_data["error"]
            return result
        
        # Agent 2: Excel Feeder
        progress_placeholder.progress(0.66)
        status_placeholder.info(f"Agent 2: Excel Feeder is mapping data from '{uploaded_file.name}' to Excel...")
        
        excel_feeder = ExcelFeederAgent()
        excel_data = excel_feeder.map_data_to_excel(extracted_data)
        result["excel_data"] = excel_data
        
        if "error" in excel_data:
            result["status"] = "error"
            result["message"] = excel_data["error"]
            return result
        
        # Create Excel file
        excel_file = excel_feeder.create_excel_file(excel_data, invoice_id)
        if isinstance(excel_file, dict) and "error" in excel_file:
            result["status"] = "error"
            result["message"] = excel_file["error"]
            return result
        result["excel_file"] = excel_file
        
        # Create Tally XML
        tally_xml = excel_feeder.create_tally_import_format(extracted_data)
        if isinstance(tally_xml, dict) and "error" in tally_xml:
            result["status"] = "error"
            result["message"] = tally_xml["error"]
            return result
        result["tally_xml"] = tally_xml
        
        # Agent 3: Data Validator
        progress_placeholder.progress(1.0)
        status_placeholder.info(f"Agent 3: Data Validator is verifying data from '{uploaded_file.name}'...")
        
        data_validator = DataValidatorAgent()
        validation_result = data_validator.validate_data(extracted_data, excel_data)
        result["validation_result"] = validation_result
        
        # Set overall status based on validation
        result["status"] = validation_result.get("status", "unknown")
        result["message"] = validation_result.get("message", "Unknown processing status")
        
        status_placeholder.success(f"Invoice '{uploaded_file.name}' processing complete!")
        progress_placeholder.empty()
        
        return result
    except Exception as e:
        return {
            "invoice_id": invoice_id,
            "file_name": uploaded_file.name if uploaded_file else "Unknown",
            "status": "error",
            "message": f"An error occurred during processing: {str(e)}",
        }

# Function to process multiple invoices in batch
def process_batch_invoices(uploaded_files):
    """Process multiple invoices in a batch."""
    if not uploaded_files:
        st.error("No files uploaded")
        return
    
    st.markdown("<div class='agent-title'>Batch Processing Results</div>", unsafe_allow_html=True)
    batch_progress = st.progress(0)
    batch_status = st.empty()
    batch_results = []
    
    total_files = len(uploaded_files)
    batch_status.info(f"Starting batch processing of {total_files} files...")
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        batch_status.info(f"Processing file {i+1} of {total_files}: {uploaded_file.name}")
        batch_progress.progress((i) / total_files)
        
        # Create a unique invoice ID
        invoice_id = f"invoice_{int(time.time())}_{i}"
        
        # Process the invoice
        result = process_invoice(uploaded_file, invoice_id)
        
        # Store result
        batch_results.append(result)
        st.session_state.processed_invoices[invoice_id] = result
        
        # Update progress
        batch_progress.progress((i + 1) / total_files)
    
    # Mark batch processing as complete
    batch_status.success(f"Batch processing complete. Processed {total_files} files.")
    st.session_state.batch_results = batch_results
    st.session_state.processing_complete = True
    
    return batch_results

def create_zip_file(files_dict):
    """Create a zip file containing multiple files."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, file_content in files_dict.items():
            zip_file.writestr(filename, file_content)
    
    zip_buffer.seek(0)
    return zip_buffer

# Main application layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.markdown("### Upload Invoices")
    
    upload_mode = st.radio("Select upload mode:", ["Single Invoice", "Multiple Invoices"])
    
    if upload_mode == "Single Invoice":
        uploaded_file = st.file_uploader("Choose an invoice file (image or PDF)", type=["jpg", "jpeg", "png", "pdf"])
        
        if st.button("Process Invoice", disabled=uploaded_file is None):
            # Process single invoice
            with st.spinner("Processing invoice..."):
                invoice_id = f"invoice_{int(time.time())}"
                result = process_invoice(uploaded_file, invoice_id)
                st.session_state.processed_invoices[invoice_id] = result
                st.session_state.current_invoice_id = invoice_id
                st.session_state.processing_complete = True
                
                # Create Tally XML if not already created
                if "tally_xml" in result and result["tally_xml"]:
                    st.session_state.tally_xml = result["tally_xml"]
    
    else:  # Multiple Invoices
        uploaded_files = st.file_uploader("Choose invoice files (images or PDFs)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
        
        if st.button("Process All Invoices", disabled=not uploaded_files):
            # Process multiple invoices
            with st.spinner("Processing invoices..."):
                batch_results = process_batch_invoices(uploaded_files)
                
                # Combine all Tally XMLs into one
                if batch_results:
                    all_vouchers = []
                    for result in batch_results:
                        if "tally_xml" in result and result["tally_xml"]:
                            # Extract the VOUCHER element from each XML
                            voucher_match = re.search(r'<VOUCHER.*?</VOUCHER>', result["tally_xml"], re.DOTALL)
                            if voucher_match:
                                all_vouchers.append(voucher_match.group(0))
                    
                    # Create a combined XML with all vouchers
                    if all_vouchers:
                        combined_xml = f"""
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
        {"".join(all_vouchers)}
        </TALLYMESSAGE>
      </REQUESTDATA>
    </IMPORTDATA>
  </BODY>
</ENVELOPE>
                        """
                        st.session_state.tally_xml = combined_xml
    
    # Reset button to clear session state
    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export options
    if st.session_state.processing_complete:
        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown("### Export Options")
        
        if upload_mode == "Single Invoice" and st.session_state.current_invoice_id:
            result = st.session_state.processed_invoices.get(st.session_state.current_invoice_id)
            if result and "excel_file" in result:
                st.download_button(
                    label="Download Excel File",
                    data=result["excel_file"],
                    file_name=f"{st.session_state.current_invoice_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            if result and "tally_xml" in result:
                st.download_button(
                    label="Download Tally XML",
                    data=result["tally_xml"],
                    file_name=f"{st.session_state.current_invoice_id}.xml",
                    mime="application/xml"
                )
        
        elif upload_mode == "Multiple Invoices" and st.session_state.batch_results:
            # Create zip files for Excel and XML
            excel_files = {}
            xml_files = {}
            
            for result in st.session_state.batch_results:
                if "error" not in result:
# Continuation of the main application code from previous section

# Export options continued for batch processing
        elif upload_mode == "Multiple Invoices" and st.session_state.batch_results:
            # Create zip files for Excel and XML
            excel_files = {}
            xml_files = {}
            
            for result in st.session_state.batch_results:
                if "error" not in result and "excel_file" in result:
                    invoice_id = result.get("invoice_id", "unknown")
                    file_name = result.get("file_name", "invoice.pdf").split(".")[0]
                    excel_files[f"{file_name}_{invoice_id}.xlsx"] = result["excel_file"].getvalue()
                
                if "error" not in result and "tally_xml" in result:
                    invoice_id = result.get("invoice_id", "unknown")
                    file_name = result.get("file_name", "invoice.pdf").split(".")[0]
                    xml_files[f"{file_name}_{invoice_id}.xml"] = result["tally_xml"]
            
            if excel_files:
                excel_zip = create_zip_file(excel_files)
                st.download_button(
                    label="Download All Excel Files (ZIP)",
                    data=excel_zip,
                    file_name="invoice_excel_files.zip",
                    mime="application/zip"
                )
            
            if xml_files:
                xml_zip = create_zip_file(xml_files)
                st.download_button(
                    label="Download All Tally XML Files (ZIP)",
                    data=xml_zip,
                    file_name="invoice_tally_xml_files.zip",
                    mime="application/zip"
                )
            
            if st.session_state.tally_xml:
                st.download_button(
                    label="Download Combined Tally XML",
                    data=st.session_state.tally_xml,
                    file_name="combined_tally_import.xml",
                    mime="application/xml"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System architecture explanation
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.markdown("### System Architecture")
    st.markdown("""
    1. **Invoice Reader**:
       - Extracts text from invoice images/PDFs
       - Recognizes handwritten notes and annotations
       - Converts data to structured JSON format
       
    2. **Excel Feeder**:
       - Maps extracted data to Excel columns
       - Creates formatted spreadsheets
       - Generates Tally-compatible XML
       
    3. **Data Validator**:
       - Verifies data accuracy and completeness
       - Checks calculations and consistency
       - Assesses quality of handwriting recognition
       - Reports any identified issues
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Display processing results if available
    if st.session_state.processing_complete:
        if upload_mode == "Single Invoice" and st.session_state.current_invoice_id:
            result = st.session_state.processed_invoices.get(st.session_state.current_invoice_id)
            
            st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
            st.markdown(f"### Processing Results: {result.get('file_name', 'Invoice')}")
            
            # Display the uploaded file if available
            if "file_name" in result:
                st.markdown("#### Uploaded Invoice")
                file_type = result.get("file_type", "")
                if file_type.startswith('image'):
                    # Display the image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=result.get("file_name"), use_column_width=True)
                elif file_type == 'application/pdf':
                    # Display PDF using HTML iframe
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    with open(tmp_file_path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    os.unlink(tmp_file_path)  # Clean up temp file
            
            # Display tabs for different results
            tabs = ["Extracted Data", "Excel Preview", "Validation Results", "Tally XML"]
            
            # Create tab buttons
            st.write("")
            cols = st.columns(len(tabs))
            active_tab = st.session_state.get("active_tab", "Extracted Data")
            
            for i, tab in enumerate(tabs):
                with cols[i]:
                    if st.button(tab, key=f"tab_{tab}"):
                        active_tab = tab
                        st.session_state.active_tab = tab
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Display tab content
            if active_tab == "Extracted Data":
                if "extracted_data" in result:
                    extracted_data = result["extracted_data"]
                    if "error" not in extracted_data:
                        # Special highlighting for handwritten notes if present
                        if "handwritten_notes" in extracted_data and extracted_data["handwritten_notes"]:
                            st.markdown(f"#### Handwritten Notes Detected")
                            st.markdown(f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>{extracted_data['handwritten_notes']}</div>", unsafe_allow_html=True)
                        
                        # Show the extracted data
                        st.markdown("#### Extracted Invoice Data")
                        st.json(extracted_data)
                    else:
                        st.error(f"Error in extraction: {extracted_data['error']}")
            
            elif active_tab == "Excel Preview":
                if "excel_data" in result:
                    excel_data = result["excel_data"]
                    if "error" not in excel_data:
                        st.markdown("#### Main Invoice Information")
                        st.dataframe(excel_data["main_info"], use_container_width=True)
                        
                        st.markdown("#### Line Items")
                        st.dataframe(excel_data["line_items"], use_container_width=True)
                    else:
                        st.error(f"Error in Excel mapping: {excel_data['error']}")
            
            elif active_tab == "Validation Results":
                if "validation_result" in result:
                    validation_result = result["validation_result"]
                    
                    # Show data quality score as a gauge
                    data_quality_score = validation_result.get("data_quality_score", 0)
                    confidence = validation_result.get("confidence", 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Data Quality Score")
                        st.progress(data_quality_score/100)
                        st.markdown(f"Score: {data_quality_score}/100")
                    
                    with col2:
                        st.markdown("#### Confidence Level")
                        st.progress(confidence/100)
                        st.markdown(f"Confidence: {confidence}/100")
                    
                    # Display overall status
                    status_class = "success" if validation_result.get("status") == "success" else "error"
                    st.markdown(f"<div class='status {status_class}'>{validation_result.get('message', 'Validation complete')}</div>", unsafe_allow_html=True)
                    
                    # Display handwriting assessment
                    handwriting_assessment = validation_result.get("handwriting_assessment", "")
                    if handwriting_assessment:
                        st.markdown("#### Handwriting Assessment")
                        st.markdown(f"<div style='background-color: #e7f3fe; padding: 10px; border-radius: 5px;'>{handwriting_assessment}</div>", unsafe_allow_html=True)
                    
                    # Display incidents if any
                    incidents = validation_result.get("incidents", [])
                    if incidents:
                        st.markdown("#### Validation Issues")
                        for incident in incidents:
                            st.markdown(f"- {incident}")
                    else:
                        st.markdown("#### No validation issues found")
            
            elif active_tab == "Tally XML":
                if "tally_xml" in result:
                    tally_xml = result["tally_xml"]
                    if isinstance(tally_xml, str):
                        st.markdown("#### Tally Prime Import XML")
                        st.code(tally_xml, language="xml")
                        
                        st.markdown("#### How to Import into Tally Prime")
                        st.markdown("""
                        1. Download the XML file using the "Download Tally XML" button
                        2. Open Tally Prime software
                        3. Navigate to: Gateway of Tally > Import Data > XML Format
                        4. Select the downloaded XML file
                        5. Verify the data and click Import
                        """)
                    else:
                        st.error("Tally XML generation failed")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif upload_mode == "Multiple Invoices" and st.session_state.batch_results:
            st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
            st.markdown("### Batch Processing Results")
            
            # Create a summary table
            summary_data = []
            for result in st.session_state.batch_results:
                file_name = result.get("file_name", "Unknown")
                status = result.get("status", "Unknown")
                message = result.get("message", "")
                
                # Get data quality score and confidence if available
                data_quality_score = "N/A"
                confidence = "N/A"
                if "validation_result" in result:
                    data_quality_score = result["validation_result"].get("data_quality_score", "N/A")
                    confidence = result["validation_result"].get("confidence", "N/A")
                
                summary_data.append({
                    "File Name": file_name,
                    "Status": status,
                    "Data Quality": data_quality_score,
                    "Confidence": confidence,
                    "Message": message[:50] + "..." if len(message) > 50 else message
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Display individual invoice details
            st.markdown("### Invoice Details")
            
            # Create an expandable section for each invoice
            for i, result in enumerate(st.session_state.batch_results):
                file_name = result.get("file_name", f"Invoice {i+1}")
                status = result.get("status", "Unknown")
                
                # Set the expander color based on status
                status_color = "#4CAF50" if status == "success" else "#F44336" if status == "error" else "#2196F3"
                st.markdown(f"<div style='border-left: 4px solid {status_color}; padding-left: 10px;'>", unsafe_allow_html=True)
                
                with st.expander(f"{file_name} - Status: {status.title()}"):
                    if "extracted_data" in result:
                        # Check for handwritten notes
                        extracted_data = result["extracted_data"]
                        if not isinstance(extracted_data, dict):
                            st.error("Invalid extracted data format")
                        else:
                            if "handwritten_notes" in extracted_data and extracted_data["handwritten_notes"]:
                                st.markdown("**Handwritten Notes:**")
                                st.markdown(f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>{extracted_data['handwritten_notes']}</div>", unsafe_allow_html=True)
                            
                            # Display extracted data summary
                            st.markdown("**Extracted Data Summary:**")
                            summary_items = {
                                "Invoice Number": extracted_data.get("invoice_number", "N/A"),
                                "Date": extracted_data.get("date", "N/A"),
                                "Vendor": extracted_data.get("vendor_name", "N/A"),
                                "Total Amount": extracted_data.get("total", "N/A"),
                                "Line Items": len(extracted_data.get("line_items", []))
                            }
                            for key, value in summary_items.items():
                                st.markdown(f"- **{key}:** {value}")
                        
                    if "validation_result" in result:
                        validation_result = result["validation_result"]
                        st.markdown("**Validation Summary:**")
                        
                        # Show data quality and confidence as horizontal bars
                        data_quality_score = validation_result.get("data_quality_score", 0)
                        confidence = validation_result.get("confidence", 0)
                        
                        data_quality_col, confidence_col = st.columns(2)
                        with data_quality_col:
                            st.markdown(f"Data Quality: {data_quality_score}/100")
                            st.progress(data_quality_score/100)
                        
                        with confidence_col:
                            st.markdown(f"Confidence: {confidence}/100")
                            st.progress(confidence/100)
                        
                        # Show incidents if any
                        incidents = validation_result.get("incidents", [])
                        if incidents:
                            st.markdown("**Issues:**")
                            for incident in incidents:
                                st.markdown(f"- {incident}")
                        else:
                            st.markdown("**No validation issues found**")
                    
                    # Add buttons for individual file downloads
                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        if "excel_file" in result:
                            st.download_button(
                                label="Download Excel",
                                data=result["excel_file"],
                                file_name=f"{file_name.split('.')[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"excel_{i}"
                            )
                    
                    with download_col2:
                        if "tally_xml" in result:
                            st.download_button(
                                label="Download Tally XML",
                                data=result["tally_xml"],
                                file_name=f"{file_name.split('.')[0]}.xml",
                                mime="application/xml",
                                key=f"xml_{i}"
                            )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Tally Prime Integration guide
            st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
            st.markdown("### Tally Prime Integration Guide")
            st.markdown("""
            #### Steps to Import XML into Tally Prime
            
            1. **Download the XML file(s)** using the "Download Combined Tally XML" button for batch import, or individual XML files for specific invoices.
            
            2. **Open Tally Prime software** on your system.
            
            3. **Navigate to Import Data**:
               - Go to Gateway of Tally
               - Select Import Data
               - Choose XML Format
            
            4. **Select the downloaded XML file** when prompted.
            
            5. **Verify the data** that Tally Prime shows in the preview.
            
            6. **Click Import** to complete the process.
            
            7. **Verify** that all vouchers have been correctly imported by checking your sales register.
            
            #### Troubleshooting
            
            - Ensure all ledgers and stock items mentioned in the XML already exist in your Tally Prime company.
            - If you encounter any errors, check that your XML file is not corrupted.
            - For multiple vouchers import, use the combined XML file rather than individual files.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown("### Instructions")
        st.markdown("""
        #### Getting Started
        
        1. **Select Upload Mode**:
           - Choose "Single Invoice" for processing one document at a time
           - Choose "Multiple Invoices" for batch processing
        
        2. **Upload Invoice Files**:
           - Supported formats: JPG, JPEG, PNG, and PDF
           - The system can process both printed and handwritten content
        
        3. **Process the Invoice(s)**:
           - Click the "Process Invoice" or "Process All Invoices" button
           - The system will analyze your documents using three specialized AI agents
        
        4. **Review Results**:
           - Examine the extracted data for accuracy
           - Check for any validation warnings or errors
           - Review how handwritten notes are captured and interpreted
        
        5. **Export Data**:
           - Download processed data as Excel files
           - Download Tally-compatible XML files for direct import
           - For batch processing, you can download individual files or combined archives
        
        #### Tips for Best Results
        
        - Ensure invoices are clearly visible and well-lit in images
        - For handwritten content, make sure writing is legible
        - PDFs work best when they contain searchable text
        - When processing multiple invoices, group similar formats together for better results
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample images
        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown("### Example Invoice Types Supported")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Standard Printed Invoice")
            st.markdown("""
            ```
            INVOICE #INV-123456
            Date: 2025-01-15
            
            Vendor: ABC Company Inc.
            
            Item        Qty   Unit Price   Total
            ----------------------------------
            Widget A    5     $10.00       $50.00
            Service B   2     $75.00       $150.00
            Part C      10    $5.50        $55.00
            
            Subtotal:               $255.00
            Tax (8%):               $20.40
            Total:                  $275.40
            ```
            """)
        
        with col2:
            st.markdown("#### Invoice with Handwritten Notes")
            st.markdown("""
            ```
            INVOICE #INV-123456
            Date: 2025-01-15
            
            Vendor: ABC Company Inc.
            
            Item        Qty   Unit Price   Total
            ----------------------------------
            Widget A    5     $10.00       $50.00
            Service B   2     $75.00       $150.00
            Part C      10    $5.50        $55.00
            
            Subtotal:               $255.00
            Tax (8%):               $20.40
            Total:                  $275.40
            
            [Handwritten] Approved for payment
            [Handwritten] Rush order - priority shipping
            ```
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Advanced Invoice Processing System | Built with Streamlit, Google Gemini, and Tally Prime Integration")
