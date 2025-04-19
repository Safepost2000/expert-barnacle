__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import io
import re
import json
import base64
import pandas as pd
import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import tempfile
import openpyxl

# Load environment variables and configure API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Automated Invoice Processing",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>Automated Invoice Processing System</div>", unsafe_allow_html=True)
st.markdown("""
This application uses three AI agents to process invoice documents:
1. **Invoice Reader**: Extracts data from invoice images/PDFs
2. **Excel Feeder**: Maps and enters data into Excel spreadsheets
3. **Data Validator**: Verifies data accuracy and reports any issues
""")

# Initialize session state variables if they don't exist
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'validation_result' not in st.session_state:
    st.session_state.validation_result = None
if 'incidents' not in st.session_state:
    st.session_state.incidents = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Initialize Gemini models
def initialize_model(model_name="gemini-2.0-flash"):
    """Initialize and return the specified Gemini model."""
    model = genai.GenerativeModel(model_name)
    return model

def initialize_text_model(model_name="gemini-2.0-flash"):
    """Initialize and return the text-only Gemini model."""
    model = genai.GenerativeModel(model_name)
    return model

# AGENT 1: Invoice Reader
class InvoiceReaderAgent:
    def __init__(self):
        self.model = initialize_model("gemini-2.0-flash")
        self.name = "Invoice Reader"
    
    def extract_data(self, image_bytes, image_type):
        """Extract invoice information from the uploaded document."""
        system_prompt = """
        You are an expert invoice data extraction agent. 
        Analyze the provided invoice image and extract the following information in JSON format:
        1. Invoice Number
        2. Date
        3. Vendor Name
        4. Line Items (as an array of objects with description, quantity, unit_price, and total)
        5. Subtotal
        6. Tax
        7. Total Amount
        
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
            "total": number
        }
        """
        
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
            return extracted_data
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

# AGENT 2: Excel Feeder
class ExcelFeederAgent:
    def __init__(self):
        self.model = initialize_text_model("gemini-2.0-flash")
        self.name = "Excel Feeder"
    
    def map_data_to_excel(self, extracted_data):
        """Map the extracted data to Excel format."""
        try:
            if "error" in extracted_data:
                return {"error": extracted_data["error"]}
            
            # Create a DataFrame for the main invoice information
            main_info = {
                "Field": ["Invoice Number", "Date", "Vendor Name", "Subtotal", "Tax", "Total Amount"],
                "Value": [
                    extracted_data.get("invoice_number", ""),
                    extracted_data.get("date", ""),
                    extracted_data.get("vendor_name", ""),
                    extracted_data.get("subtotal", ""),
                    extracted_data.get("tax", ""),
                    extracted_data.get("total", "")
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
    
    def create_excel_file(self, excel_data):
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

# AGENT 3: Data Validator
class DataValidatorAgent:
    def __init__(self):
        self.model = initialize_text_model("gemini-2.0-flash")
        self.name = "Data Validator"
    
    def validate_data(self, extracted_data, excel_data):
        """Validate the extracted data against the original invoice."""
        if "error" in extracted_data or "error" in excel_data:
            error_message = extracted_data.get("error", "") or excel_data.get("error", "")
            return {"status": "error", "message": error_message, "incidents": [error_message]}
        
        system_prompt = """
        You are a data validation expert. Analyze the extracted invoice data and identify any potential issues or inconsistencies.
        
        Check the following:
        1. Are all required fields present? (Invoice Number, Date, Vendor Name, Line Items, Subtotal, Tax, Total Amount)
        2. Are the numerical calculations correct? (Sum of line items should equal subtotal, subtotal + tax should equal total)
        3. Are there any suspicious or unusual values?
        
        Return your analysis in JSON format with the following structure:
        {
            "status": "success" or "error",
            "message": "Your overall assessment",
            "incidents": [
                "Description of issue 1",
                "Description of issue 2"
            ]
        }
        
        If no issues are found, return:
        {
            "status": "success",
            "message": "Invoice processed with no errors.",
            "incidents": []
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
                "incidents": [f"Validation failed: {str(e)}"]
            }

# Function to process the invoice
def process_invoice():
    """Process the uploaded invoice using the three-agent system."""
    if uploaded_file is None:
        st.error("Please upload an invoice file.")
        return
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Display the uploaded file
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Invoice", use_container_width=True)
            image_bytes = uploaded_file.getvalue()
            image_type = uploaded_file.type
        elif uploaded_file.type == 'application/pdf':
            # Save the PDF temporarily and display using HTML iframe
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Display PDF
            with open(tmp_file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            image_bytes = uploaded_file.getvalue()
            image_type = uploaded_file.type
            os.unlink(tmp_file_path)  # Clean up temp file
        else:
            st.error("Unsupported file type. Please upload an image or PDF.")
            return
        
        # Agent 1: Invoice Reader
        progress_placeholder.progress(0.33)
        status_placeholder.info("Agent 1: Invoice Reader is extracting data...")
        
        invoice_reader = InvoiceReaderAgent()
        extracted_data = invoice_reader.extract_data(image_bytes, image_type)
        st.session_state.extracted_data = extracted_data
        
        if "error" in extracted_data:
            st.error(f"Error in Invoice Reader: {extracted_data['error']}")
            return
        
        # Display extracted data
        st.markdown("<div class='agent-title'>Agent 1: Invoice Reader - Extracted Data</div>", unsafe_allow_html=True)
        st.json(extracted_data)
        
        # Agent 2: Excel Feeder
        progress_placeholder.progress(0.66)
        status_placeholder.info("Agent 2: Excel Feeder is mapping data to Excel...")
        
        excel_feeder = ExcelFeederAgent()
        excel_data = excel_feeder.map_data_to_excel(extracted_data)
        st.session_state.excel_data = excel_data
        
        if "error" in excel_data:
            st.error(f"Error in Excel Feeder: {excel_data['error']}")
            return
        
        # Display mapped data
        st.markdown("<div class='agent-title'>Agent 2: Excel Feeder - Excel Data</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Invoice Information**")
            st.dataframe(excel_data["main_info"])
        with col2:
            st.markdown("**Line Items**")
            st.dataframe(excel_data["line_items"])
        
        # Generate and offer Excel download
        excel_file = excel_feeder.create_excel_file(excel_data)
        if isinstance(excel_file, dict) and "error" in excel_file:
            st.error(f"Error creating Excel file: {excel_file['error']}")
        else:
            st.download_button(
                label="Download Excel File",
                data=excel_file,
                file_name="processed_invoice.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Agent 3: Data Validator
        progress_placeholder.progress(1.0)
        status_placeholder.info("Agent 3: Data Validator is verifying data...")
        
        data_validator = DataValidatorAgent()
        validation_result = data_validator.validate_data(extracted_data, excel_data)
        st.session_state.validation_result = validation_result
        st.session_state.incidents = validation_result.get("incidents", [])
        
        # Display validation results
        st.markdown("<div class='agent-title'>Agent 3: Data Validator - Validation Results</div>", unsafe_allow_html=True)
        status_class = "success" if validation_result["status"] == "success" else "error"
        st.markdown(f"<div class='status {status_class}'>{validation_result['message']}</div>", unsafe_allow_html=True)
        
        if validation_result["incidents"]:
            st.markdown("**Incidents:**")
            for incident in validation_result["incidents"]:
                st.markdown(f"- {incident}")
        
        # Set processing complete flag
        st.session_state.processing_complete = True
        status_placeholder.success("Invoice processing complete!")
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        progress_placeholder.empty()
        status_placeholder.empty()

# Main application layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.markdown("### Upload Invoice")
    uploaded_file = st.file_uploader("Choose an invoice file (image or PDF)", type=["jpg", "jpeg", "png", "pdf"])
    
    if st.button("Process Invoice", disabled=uploaded_file is None):
        process_invoice()
    
    # Reset button to clear session state
    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show system architecture
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.markdown("### System Architecture")
    st.markdown("""
    1. **Invoice Reader**:
       - Extracts text and data from invoice
       - Identifies key fields
       - Structures data in JSON format
       
    2. **Excel Feeder**:
       - Maps extracted data to Excel columns
       - Creates structured spreadsheet
       - Formats data appropriately
       
    3. **Data Validator**:
       - Verifies data accuracy
       - Checks calculations and consistency
       - Reports any found issues
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Display processing results if available
    if st.session_state.processing_complete:
        pass  # Results are displayed in the process_invoice function
    else:
        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload an invoice image or PDF using the file uploader on the left
        2. Click the "Process Invoice" button to start the automated processing
        3. The system will:
           - Extract all relevant information from the invoice
           - Map the data to Excel format
           - Validate the data for accuracy
        4. Review the results and download the Excel file
        5. Check for any incidents reported by the Data Validator
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample invoice display
        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown("### Sample Invoice Format")
        sample_invoice = """
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
        """
        st.code(sample_invoice)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Automated Invoice Processing System | Built with Streamlit and Google Gemini")
