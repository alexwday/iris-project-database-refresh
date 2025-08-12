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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER, TA_RIGHT
from reportlab.lib import colors
from reportlab.platypus import HRFlowable
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

# Excel column mapping (A-W, 0-22 in zero-based indexing)
# Column layout based on actual Excel structure:
# A: Year
# B: Month  
# C: IFRS standard
# D: US GAAP
# E: Other related IFRS/US GAAP standards
# F: If related to financial instruments state subtopic
# G: Related product
# H: Related platform
# I: Accounting question of the Issue
# J: Conclusion reached
# K: Key facts/circumstances leading to conclusion
# L: Specific guidance reference paragraph #'s
# M: IFRS/US GAAP differences identified
# N: Benchmarking
# O: Preparer
# P: Stakeholder concurrence obtained
# Q: PwC concurrence obtained
# R: Server link
# S: Key file name(s) on the server location
# T: Is CAPM update required and completed
# U: If CAPM was updated when was it published
# V: Related CAPM
# W: Initial and date of APG Senior Director reviewer

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
    """Create a professionally formatted PDF document from a single Excel row."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.5*inch,
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define professional color scheme
    PRIMARY_COLOR = colors.HexColor('#1E3A8A')  # Deep navy blue
    ACCENT_COLOR = colors.HexColor('#0D9488')   # Professional teal
    BG_COLOR = colors.HexColor('#F8FAFC')       # Light gray
    TEXT_COLOR = colors.HexColor('#374151')      # Charcoal
    LINK_COLOR = colors.HexColor('#2563EB')      # Standard blue
    
    # Define comprehensive styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=PRIMARY_COLOR,
        spaceAfter=8,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=ACCENT_COLOR,
        spaceAfter=4,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=PRIMARY_COLOR,
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        leftIndent=0,
        alignment=TA_LEFT
    )
    
    field_label_style = ParagraphStyle(
        'FieldLabel',
        parent=styles['Normal'],
        fontSize=9,
        textColor=ACCENT_COLOR,
        spaceAfter=2,
        fontName='Helvetica-Bold',
        leftIndent=0
    )
    
    field_value_style = ParagraphStyle(
        'FieldValue',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=8,
        leftIndent=0,
        textColor=TEXT_COLOR,
        fontName='Times-Roman'
    )
    
    summary_box_style = ParagraphStyle(
        'SummaryBox',
        parent=styles['Normal'],
        fontSize=10,
        textColor=TEXT_COLOR,
        fontName='Helvetica',
        leftIndent=6,
        rightIndent=6,
        alignment=TA_LEFT
    )
    
    link_style = ParagraphStyle(
        'LinkStyle',
        parent=field_value_style,
        textColor=LINK_COLOR,
        fontSize=9,
        fontName='Courier'
    )
    
    metadata_style = ParagraphStyle(
        'Metadata',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#9CA3AF'),
        alignment=TA_RIGHT
    )
    
    # Helper function to safely get value
    def get_value(col_index):
        try:
            if col_index < len(row_data):
                value = row_data.iloc[col_index]
                if pd.isna(value) or value is None:
                    return None
                value_str = str(value).strip()
                if not value_str or value_str.lower() in ['nan', 'none', 'null', 'n/a']:
                    return None
                return value_str
            return None
        except:
            return None
    
    # Helper function to format value for display
    def format_value(value):
        if value is None:
            return ""
        # Escape HTML special characters
        value = str(value).replace('&', '&amp;')
        value = value.replace('<', '&lt;')
        value = value.replace('>', '&gt;')
        value = value.replace('"', '&quot;')
        value = value.replace("'", '&#39;')
        # Handle newlines
        value = value.replace('\n', '<br/>')
        return value
    
    # Get all field values
    year = get_value(0)
    month = get_value(1)
    ifrs_standard = get_value(2)
    us_gaap = get_value(3)
    other_standards = get_value(4)
    fi_subtopic = get_value(5)
    related_product = get_value(6)
    related_platform = get_value(7)
    accounting_question = get_value(8)
    conclusion = get_value(9)
    key_facts = get_value(10)
    guidance_ref = get_value(11)
    differences = get_value(12)
    benchmarking = get_value(13)
    preparer = get_value(14)
    stakeholder_concurrence = get_value(15)
    pwc_concurrence = get_value(16)
    server_link = get_value(17)
    key_files = get_value(18)
    capm_required = get_value(19)
    capm_date = get_value(20)
    related_capm = get_value(21)
    apg_reviewer = get_value(22)
    
    # HEADER SECTION
    story.append(Paragraph("APG Wiki Entry", title_style))
    
    # Add date and entry number
    date_str = f"{month or 'N/A'} {year or 'N/A'}"
    story.append(Paragraph(f"Entry #{row_number} | {date_str}", subtitle_style))
    
    # Add reviewer info if available
    if apg_reviewer:
        story.append(Paragraph(f"Reviewed by: {format_value(apg_reviewer)}", metadata_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # EXECUTIVE SUMMARY BOX
    summary_data = []
    summary_row = []
    
    # Build standards summary
    standards = []
    if ifrs_standard:
        standards.append(f"IFRS: {ifrs_standard}")
    if us_gaap:
        standards.append(f"US GAAP: {us_gaap}")
    
    if standards or related_product or related_platform:
        summary_table = Table([
            [Paragraph("<b>Standards & Products Summary</b>", field_label_style)],
            [Paragraph(" | ".join(standards) if standards else "No standards specified", summary_box_style)],
            [Paragraph(f"Product: {related_product or 'N/A'} | Platform: {related_platform or 'N/A'}", summary_box_style)]
        ], colWidths=[6.5*inch])
        
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), BG_COLOR),
            ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
    
    # CORE ISSUE ANALYSIS SECTION
    if accounting_question or conclusion or key_facts:
        story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT_COLOR))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Core Issue Analysis", section_heading_style))
        
        # Accounting Question
        if accounting_question:
            question_table = Table([
                [Paragraph("<b>Accounting Question</b>", field_label_style)],
                [Paragraph(format_value(accounting_question), field_value_style)]
            ], colWidths=[6.5*inch])
            
            question_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ACCENT_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0FDFA')),
                ('BOX', (0, 0), (-1, -1), 0.5, ACCENT_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(question_table)
            story.append(Spacer(1, 0.15*inch))
        
        # Key Facts
        if key_facts:
            story.append(Paragraph("<b>Key Facts & Circumstances</b>", field_label_style))
            story.append(Paragraph(format_value(key_facts), field_value_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Conclusion
        if conclusion:
            conclusion_table = Table([
                [Paragraph("<b>Conclusion Reached</b>", field_label_style)],
                [Paragraph(format_value(conclusion), field_value_style)]
            ], colWidths=[6.5*inch])
            
            conclusion_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EFF6FF')),
                ('BOX', (0, 0), (-1, -1), 0.5, PRIMARY_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(conclusion_table)
        
        story.append(Spacer(1, 0.2*inch))
    
    # TECHNICAL DETAILS SECTION
    if other_standards or fi_subtopic or guidance_ref or differences or benchmarking:
        story.append(Paragraph("Technical Details & References", section_heading_style))
        
        technical_data = []
        if other_standards:
            technical_data.append([Paragraph("<b>Other Standards:</b>", field_label_style), 
                                  Paragraph(format_value(other_standards), field_value_style)])
        if fi_subtopic:
            technical_data.append([Paragraph("<b>Financial Instruments:</b>", field_label_style), 
                                  Paragraph(format_value(fi_subtopic), field_value_style)])
        if guidance_ref:
            technical_data.append([Paragraph("<b>Guidance References:</b>", field_label_style), 
                                  Paragraph(format_value(guidance_ref), field_value_style)])
        if differences:
            technical_data.append([Paragraph("<b>IFRS/US GAAP Differences:</b>", field_label_style), 
                                  Paragraph(format_value(differences), field_value_style)])
        if benchmarking:
            technical_data.append([Paragraph("<b>Benchmarking:</b>", field_label_style), 
                                  Paragraph(format_value(benchmarking), field_value_style)])
        
        if technical_data:
            technical_table = Table(technical_data, colWidths=[1.8*inch, 4.7*inch])
            technical_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (0, -1), 0),
                ('LEFTPADDING', (1, 0), (1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(technical_table)
            story.append(Spacer(1, 0.2*inch))
    
    # REVIEW & APPROVALS SECTION
    if preparer or stakeholder_concurrence or pwc_concurrence:
        story.append(Paragraph("Review & Approvals", section_heading_style))
        
        approval_data = []
        if preparer:
            approval_data.append(["Preparer", format_value(preparer)])
        if stakeholder_concurrence:
            status = "✓ Obtained" if stakeholder_concurrence.lower() in ['yes', 'y', 'true'] else format_value(stakeholder_concurrence)
            approval_data.append(["Stakeholder Concurrence", status])
        if pwc_concurrence:
            status = "✓ Obtained" if pwc_concurrence.lower() in ['yes', 'y', 'true'] else format_value(pwc_concurrence)
            approval_data.append(["PwC Concurrence", status])
        
        if approval_data:
            # Convert to Paragraph objects for the table
            approval_table_data = []
            for label, value in approval_data:
                approval_table_data.append([
                    Paragraph(f"<b>{label}:</b>", field_label_style),
                    Paragraph(value, field_value_style)
                ])
            
            approval_table = Table(approval_table_data, colWidths=[2*inch, 4.5*inch])
            approval_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
                ('BACKGROUND', (0, 0), (0, -1), BG_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(approval_table)
            story.append(Spacer(1, 0.2*inch))
    
    # DOCUMENTATION & CAPM SECTION
    if server_link or key_files or capm_required or capm_date or related_capm:
        story.append(Paragraph("Documentation & CAPM", section_heading_style))
        
        if server_link:
            story.append(Paragraph("<b>Server Link:</b>", field_label_style))
            if server_link.startswith(('http://', 'https://', '\\\\', '//')):
                link_text = f'<link href="{format_value(server_link)}" color="blue">{format_value(server_link)}</link>'
                story.append(Paragraph(link_text, link_style))
            else:
                story.append(Paragraph(format_value(server_link), field_value_style))
            story.append(Spacer(1, 0.05*inch))
        
        if key_files:
            story.append(Paragraph("<b>Key Files:</b>", field_label_style))
            story.append(Paragraph(format_value(key_files), field_value_style))
            story.append(Spacer(1, 0.05*inch))
        
        # CAPM Information
        if capm_required or capm_date or related_capm:
            capm_data = []
            if capm_required:
                capm_data.append([Paragraph("<b>CAPM Update:</b>", field_label_style), 
                                Paragraph(format_value(capm_required), field_value_style)])
            if capm_date:
                capm_data.append([Paragraph("<b>Publication Date:</b>", field_label_style), 
                                Paragraph(format_value(capm_date), field_value_style)])
            if related_capm:
                capm_data.append([Paragraph("<b>Related CAPM:</b>", field_label_style), 
                                Paragraph(format_value(related_capm), field_value_style)])
            
            if capm_data:
                capm_table = Table(capm_data, colWidths=[1.5*inch, 5*inch])
                capm_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (0, -1), 0),
                    ('LEFTPADDING', (1, 0), (1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(capm_table)
    
    # FOOTER
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E5E7EB')))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Internal APG Wiki Database", metadata_style))
    
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