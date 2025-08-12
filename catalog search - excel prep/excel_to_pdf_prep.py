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

# --- Excel Configuration ---
# Name of the specific sheet to read from the Excel file
EXCEL_SHEET_NAME = "APG WIKI"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# Excel column mapping organized by sections
# Based on actual Excel columns A-W (0-22 in zero-based indexing)
COLUMN_SECTIONS = {
    'Date Information': {
        'Year': 0,  # Column A
        'Month': 1  # Column B
    },
    'Accounting Standards': {
        'IFRS Standard': 2,  # Column C
        'US GAAP': 3,  # Column D
        'Other Related IFRS/US GAAP Standards': 4,  # Column E
        'Financial Instruments Subtopic': 5  # Column F
    },
    'Product & Platform Details': {
        'Related Product': 6,  # Column G
        'Related Platform': 7  # Column H
    },
    'Issue Analysis & Conclusion': {
        'Accounting Question/Issue': 8,  # Column I
        'Conclusion Reached': 9,  # Column J
        'Key Facts & Circumstances': 10  # Column K
    },
    'Guidance & References': {
        'Specific Guidance Reference (Paragraph #s)': 11,  # Column L
        'IFRS/US GAAP Differences Identified': 12,  # Column M
        'Benchmarking': 13  # Column N
    },
    'Review & Approval': {
        'Preparer': 14,  # Column O
        'Stakeholder Concurrence Obtained': 15,  # Column P
        'PwC Concurrence Obtained': 16,  # Column Q
        'APG Senior Director Reviewer': 22  # Column W
    },
    'Documentation & Files': {
        'Server Link': 17,  # Column R
        'Key File Name(s)': 18  # Column S
    },
    'CAPM Updates': {
        'CAPM Update Required/Completed': 19,  # Column T
        'CAPM Publication Date': 20,  # Column U
        'Related CAPM': 21  # Column V
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
        
        # Read Excel file - specifically the configured sheet
        try:
            df = pd.read_excel(file_obj, engine='openpyxl', sheet_name=EXCEL_SHEET_NAME)
            print(f"Successfully read sheet '{EXCEL_SHEET_NAME}' from Excel file: {file_path} ({len(df)} rows)")
        except ValueError as ve:
            # Sheet doesn't exist - list available sheets for debugging
            file_obj.seek(0)  # Reset buffer position
            excel_file = pd.ExcelFile(file_obj, engine='openpyxl')
            available_sheets = excel_file.sheet_names
            print(f"[ERROR] Sheet '{EXCEL_SHEET_NAME}' not found in Excel file.")
            print(f"Available sheets: {', '.join(available_sheets)}")
            return None
        
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
    
    # Define styles for professional formatting
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading1'],
        fontSize=13,
        textColor=colors.HexColor('#003366'),
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=0,
        borderColor=colors.HexColor('#003366'),
        borderRadius=0
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=10,
        textColor=colors.HexColor('#444444'),
        spaceAfter=3,
        fontName='Helvetica-Bold',
        leftIndent=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=10,
        leftIndent=24,
        textColor=colors.HexColor('#333333')
    )
    
    metadata_style = ParagraphStyle(
        'Metadata',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=3
    )
    
    # Helper function to format value
    def format_value(value):
        if pd.isna(value) or value is None:
            return "N/A"
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['nan', 'none', 'null']:
            return "N/A"
        # Escape HTML special characters for reportlab
        value_str = value_str.replace('&', '&amp;')
        value_str = value_str.replace('<', '&lt;')
        value_str = value_str.replace('>', '&gt;')
        value_str = value_str.replace('"', '&quot;')
        value_str = value_str.replace("'", '&#39;')
        # Handle newlines - preserve formatting
        value_str = value_str.replace('\n', '<br/>')
        # Handle multiple spaces
        value_str = value_str.replace('  ', '&nbsp;&nbsp;')
        return value_str
    
    # Add title with entry number
    story.append(Paragraph(f"APG Wiki Entry #{row_number}", title_style))
    
    # Add metadata section
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", metadata_style))
    story.append(Paragraph(f"Source: Internal APG Wiki Database", metadata_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Add a horizontal line for visual separation
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 0.2*inch))
    
    # Process each section with improved formatting
    for section_name, fields in COLUMN_SECTIONS.items():
        # Check if section has any non-empty values
        has_content = False
        for field_name, col_index in fields.items():
            try:
                value = row_data.iloc[col_index] if col_index < len(row_data) else None
                if value and not pd.isna(value) and str(value).strip():
                    has_content = True
                    break
            except:
                pass
        
        # Only add section if it has content
        if has_content:
            # Add section heading with underline effect
            story.append(Paragraph(f"<u>{section_name}</u>", section_style))
            
            # Process fields in this section
            for field_name, col_index in fields.items():
                # Get value from row safely
                try:
                    value = row_data.iloc[col_index] if col_index < len(row_data) else None
                except (IndexError, KeyError):
                    value = None
                    print(f"[WARNING] Could not access column {col_index} for field '{field_name}'.")
                
                # Only add field if it has content
                value_str = format_value(value)
                if value_str != "Not specified" and value_str != "N/A":
                    # Add field heading and content
                    story.append(Paragraph(f"<b>{field_name}:</b>", heading_style))
                    
                    # Special formatting for certain fields
                    if field_name in ['Server Link'] and value_str not in ["Not specified", "N/A"]:
                        # Format as clickable link if it looks like a URL or path
                        if value_str.startswith(('http://', 'https://', '\\\\', '//')):
                            value_str = f'<link href="{value_str}" color="blue">{value_str}</link>'
                    
                    story.append(Paragraph(value_str, body_style))
            
            story.append(Spacer(1, 0.15*inch))
    
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
        
        # We use positional indices (0-22 for columns A-W), so we need at least 23 columns
        # Extra columns at the end don't matter since we won't access them
        MIN_REQUIRED_COLUMNS = 23
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