# -*- coding: utf-8 -*-
"""
Excel to PDF Preprocessor for Catalog Search Pipeline

This script reads an Excel file from NAS containing internal wiki data,
converts each row into a formatted PDF document, and saves the PDFs
back to NAS for processing by the catalog search stage 1 process.

Each Excel row is converted to a nicely formatted PDF with proper 
field labels and content organized into logical sections.
"""

import pandas as pd
import sys
import os
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
import socket
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import openpyxl

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- NAS Configuration ---
# Network attached storage connection parameters
# NOTE: Update these values to match your actual NAS configuration
# These should match the configuration in catalog search/stage1_extract_csv.py
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445
}

# Input path on NAS where Excel file is located (relative to share)
# This folder should contain exactly one Excel file (.xlsx or .xls)
NAS_INPUT_FOLDER_PATH = "path/to/excel/input/folder"  # e.g., "iris/internal_wiki/input"

# Output path on NAS where PDFs will be saved (relative to share)
# PDFs will be named "APG Wiki - Row X.pdf" where X is the Excel row number
NAS_OUTPUT_FOLDER_PATH = "path/to/pdf/output/folder"  # e.g., "iris/internal_wiki/output"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# Excel column mapping organized by sections
COLUMN_SECTIONS = {
    'Record Date': {
        'Year': 0,
        'Month': 1
    },
    'Standards and Classification': {
        'Standards and Classification': 2,
        'IFRS Standards': 3,
        'US GAAP': 4,
        'Other Related Standards': 5,
        'Financial Instruments Topics': 6,
        'Related Product': 7,
        'Related Platform': 8
    },
    'Wiki Details': {
        'Issue Analysis': 9,
        'Question': 10,
        'Conclusion': 11,
        'Key Facts/Circumstances': 12
    },
    'References and Documentation': {
        'References and Documentation': 13,
        'Guidance Reference': 14,
        'IFRS/US GAAP Differences': 15,
        'Benchmarking': 16,
        'Server Link': 17,
        'Key Files': 18,
        'Related CAPM': 19
    },
    'Review and Approval Information': {
        'Review and Approval Information': 20,
        'Preparer': 21,
        'Stakeholder Concurrence': 22,
        'PwC Concurrence': 23,
        'APG Director Review': 24,
        'CAPM Update Required': 25,
        'CAPM Update Date': 26
    }
}

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def create_nas_connection():
    """Creates and returns an authenticated SMBConnection object."""
    try:
        conn = SMBConnection(
            NAS_PARAMS["user"],
            NAS_PARAMS["password"],
            CLIENT_HOSTNAME,
            NAS_PARAMS["ip"],
            use_ntlm_v2=True,
            is_direct_tcp=(NAS_PARAMS["port"] == 445)
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60)
        if not connected:
            print("[ERROR] Failed to connect to NAS.")
            return None
        print(f"Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']} on share '{NAS_PARAMS['share']}'")
        return conn
    except Exception as e:
        print(f"[ERROR] Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        print("[ERROR] Cannot ensure NAS directory: No connection.")
        return False
    
    path_parts = dir_path.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                conn.listPath(share_name, current_path)
            except Exception:
                print(f"Creating directory on NAS: {current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to ensure/create NAS directory '{dir_path}': {e}")
        return False

def list_excel_files_from_nas(conn, share_name, folder_path):
    """List all Excel files in the specified NAS folder."""
    excel_files = []
    try:
        items = conn.listPath(share_name, folder_path)
        for item in items:
            if item.filename == '.' or item.filename == '..':
                continue
            if not item.isDirectory and (item.filename.endswith('.xlsx') or item.filename.endswith('.xls')):
                excel_files.append(os.path.join(folder_path, item.filename).replace('\\', '/'))
        return excel_files
    except Exception as e:
        print(f"[ERROR] Failed to list Excel files from NAS: {e}")
        return []

def read_excel_from_nas(conn, share_name, file_path):
    """Read an Excel file from NAS into a pandas DataFrame."""
    try:
        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, file_path, file_obj)
        file_obj.seek(0)
        
        # Read Excel file
        df = pd.read_excel(file_obj, engine='openpyxl')
        print(f"Successfully read Excel file from NAS: {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read Excel file from NAS '{file_path}': {e}")
        return None

def create_pdf_from_row(row_data, row_number):
    """Create a PDF document from a single Excel row."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#003366'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#555555'),
        spaceAfter=4,
        fontName='Helvetica-Bold',
        leftIndent=15
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leftIndent=30
    )
    
    # Helper function to format value
    def format_value(value):
        if pd.isna(value) or value is None:
            return "Not specified"
        value_str = str(value).strip()
        if not value_str:
            return "Not specified"
        # Escape HTML special characters for reportlab
        value_str = value_str.replace('&', '&amp;')
        value_str = value_str.replace('<', '&lt;')
        value_str = value_str.replace('>', '&gt;')
        value_str = value_str.replace('"', '&quot;')
        value_str = value_str.replace("'", '&#39;')
        # Handle newlines
        value_str = value_str.replace('\n', '<br/>')
        return value_str
    
    # Add title
    story.append(Paragraph(f"APG Wiki - Row {row_number}", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add metadata
    story.append(Paragraph(f"<font size=9>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font>", styles['Normal']))
    story.append(Paragraph(f"<font size=9>Source: Internal Wiki Excel Data</font>", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Process each section
    for section_name, fields in COLUMN_SECTIONS.items():
        # Add section heading
        story.append(Paragraph(section_name, section_style))
        
        # Process fields in this section
        for field_name, col_index in fields.items():
            # Skip section title fields (they're redundant with section headers)
            if field_name in ['Standards and Classification', 'References and Documentation', 'Review and Approval Information']:
                continue
            
            # Get value from row safely
            try:
                value = row_data.iloc[col_index] if col_index < len(row_data) else None
            except (IndexError, KeyError):
                value = None
                print(f"[WARNING] Could not access column {col_index} for field '{field_name}'.")
            
            value_str = format_value(value)
            
            # Add field heading and content
            story.append(Paragraph(field_name, heading_style))
            story.append(Paragraph(value_str, body_style))
        
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def save_pdf_to_nas(conn, share_name, file_path, pdf_content):
    """Save a PDF to NAS."""
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            print(f"[ERROR] Failed to ensure output directory exists: {dir_path}")
            return False
        
        # Write PDF to NAS
        file_obj = io.BytesIO(pdf_content)
        bytes_written = conn.storeFile(share_name, file_path, file_obj)
        print(f"Successfully wrote PDF ({bytes_written} bytes) to: {file_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write PDF to NAS '{file_path}': {e}")
        return False

# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("Excel to PDF Preprocessor for Catalog Search Pipeline")
    print("="*60 + "\n")
    
    # Connect to NAS
    print("[1] Connecting to NAS...")
    conn = create_nas_connection()
    if not conn:
        print("[CRITICAL ERROR] Failed to connect to NAS. Exiting.")
        sys.exit(1)
    
    try:
        # List Excel files in input folder
        print(f"\n[2] Searching for Excel files in: {NAS_PARAMS['share']}/{NAS_INPUT_FOLDER_PATH}")
        excel_files = list_excel_files_from_nas(conn, NAS_PARAMS["share"], NAS_INPUT_FOLDER_PATH)
        
        if not excel_files:
            print("[ERROR] No Excel files found in input folder.")
            sys.exit(1)
        
        if len(excel_files) > 1:
            print(f"[ERROR] Multiple Excel files found ({len(excel_files)}). Only one file expected.")
            print("Files found:")
            for f in excel_files:
                print(f"  - {f}")
            sys.exit(1)
        
        excel_file_path = excel_files[0]
        print(f"Found Excel file: {excel_file_path}")
        
        # Read Excel file
        print(f"\n[3] Reading Excel file from NAS...")
        df = read_excel_from_nas(conn, NAS_PARAMS["share"], excel_file_path)
        
        if df is None or df.empty:
            print("[ERROR] Failed to read Excel file or file is empty.")
            sys.exit(1)
        
        print(f"Excel file loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # We use positional indices (0-26), so we only need to ensure we have enough columns
        # Extra columns at the end don't matter since we won't access them
        MIN_REQUIRED_COLUMNS = 27
        if len(df.columns) < MIN_REQUIRED_COLUMNS:
            print(f"[WARNING] Excel has {len(df.columns)} columns, expected at least {MIN_REQUIRED_COLUMNS}")
            print("Some fields may be missing in the generated PDFs")
        
        # Check if first row is header
        is_header_row = False
        if len(df) > 0:
            first_cell = str(df.iloc[0, 0]).lower()
            # Also check other expected header values to be more certain
            second_cell = str(df.iloc[0, 1]).lower() if len(df.columns) > 1 else ""
            if first_cell == 'year' or (first_cell == 'year' and second_cell == 'month'):
                is_header_row = True
                print("First row appears to be a header row - will skip it")
        
        # Process each row
        print(f"\n[4] Converting rows to PDF documents...")
        success_count = 0
        error_count = 0
        
        for index, row in df.iterrows():
            # Skip header row if it exists
            if index == 0 and is_header_row:
                print(f"Skipping header row")
                continue
            
            # Calculate actual Excel row number:
            # - Excel uses 1-based indexing (not 0-based like Python)
            # - If there's a header row, we need to account for it
            # Example: If header at row 1, first data row (index 1) is Excel row 2
            excel_row_number = index + 2 if is_header_row else index + 1
            
            print(f"\nProcessing row {excel_row_number}...")
            
            try:
                # Create PDF from row
                pdf_content = create_pdf_from_row(row, excel_row_number)
                
                # Save PDF to NAS
                pdf_filename = f"APG Wiki - Row {excel_row_number}.pdf"
                pdf_path = os.path.join(NAS_OUTPUT_FOLDER_PATH, pdf_filename).replace('\\', '/')
                
                if save_pdf_to_nas(conn, NAS_PARAMS["share"], pdf_path, pdf_content):
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"[ERROR] Failed to process row {excel_row_number}: {e}")
                error_count += 1
        
        # Summary
        print("\n" + "="*60)
        print("Processing Complete!")
        print(f"Successfully converted: {success_count} rows")
        if error_count > 0:
            print(f"Failed conversions: {error_count} rows")
        print("="*60 + "\n")
        
    finally:
        if conn:
            conn.close()
            print("NAS connection closed.")

if __name__ == "__main__":
    main()