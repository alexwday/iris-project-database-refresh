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
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
                                Table, TableStyle, KeepTogether, HRFlowable, 
                                FrameBreak, KeepInFrame, Flowable, PageTemplate,
                                Frame, BaseDocTemplate, NextPageTemplate)
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
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

# ==============================================================================
# --- Custom Document Template with Headers and Footers ---
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """Custom canvas to add headers and footers to each page."""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        self.row_number = None
        self.month = None
        self.year = None
        self.generation_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_header_footer(self, page_count):
        """Draw header and footer on each page."""
        # Save the state
        self.saveState()
        
        # Define colors
        primary_dark = colors.HexColor('#1e293b')
        text_secondary = colors.HexColor('#6b7280')
        border_color = colors.HexColor('#e5e7eb')
        
        # Header
        self.setFont("Helvetica-Bold", 10)
        self.setFillColor(primary_dark)
        self.drawString(0.5*inch, letter[1] - 0.35*inch, "APG Wiki")
        
        self.setFont("Helvetica", 9)
        self.setFillColor(text_secondary)
        date_str = f"{self.month or ''} {self.year or ''}"
        header_right = f"Row #{self.row_number} | {date_str}"
        self.drawRightString(letter[0] - 0.5*inch, letter[1] - 0.35*inch, header_right)
        
        # Header line
        self.setStrokeColor(border_color)
        self.setLineWidth(0.5)
        self.line(0.5*inch, letter[1] - 0.45*inch, letter[0] - 0.5*inch, letter[1] - 0.45*inch)
        
        # Footer
        self.setFont("Helvetica", 7)
        self.setFillColor(text_secondary)
        footer_text = f"Generated: {self.generation_time} | Internal APG Wiki Database"
        
        # Footer line
        self.line(0.5*inch, 0.45*inch, letter[0] - 0.5*inch, 0.45*inch)
        
        # Footer text - centered
        text_width = self.stringWidth(footer_text, "Helvetica", 7)
        x_centered = (letter[0] - text_width) / 2
        self.drawString(x_centered, 0.3*inch, footer_text)
        
        # Page number (right aligned)
        page_num = self.pages.index(self.__dict__) + 1
        if page_count > 1:
            self.drawRightString(letter[0] - 0.5*inch, 0.3*inch, f"Page {page_num} of {page_count}")
        
        # Restore the state
        self.restoreState()

class APGWikiDocTemplate(BaseDocTemplate):
    """Custom document template with headers and footers."""
    def __init__(self, filename, row_number, month, year, **kwargs):
        self.row_number = row_number
        self.month = month
        self.year = year
        BaseDocTemplate.__init__(self, filename, **kwargs)
        
        # Define the page template with proper margins for headers/footers
        frame = Frame(
            self.leftMargin, 
            self.bottomMargin + 0.3*inch,  # Leave space for footer
            self.width, 
            self.height - 0.5*inch,  # Leave space for header and footer
            id='normal'
        )
        
        template = PageTemplate('normal', [frame])
        self.addPageTemplates([template])
    
    def build(self, flowables):
        """Build the document with custom canvas."""
        self._calc()
        BaseDocTemplate.build(self, flowables, canvasmaker=self._make_canvas)
    
    def _make_canvas(self, *args, **kwargs):
        """Create custom canvas with row information."""
        c = NumberedCanvas(*args, **kwargs)
        c.row_number = self.row_number
        c.month = self.month
        c.year = self.year
        return c

# ==============================================================================
# --- Container Continuation Flowable ---
# ==============================================================================

class ContinuationTitle(Flowable):
    """A flowable that creates a continuation title for split containers."""
    def __init__(self, title, section_header_style, header_color, COLORS):
        Flowable.__init__(self)
        self.title = title
        self.section_header_style = section_header_style
        self.header_color = header_color
        self.COLORS = COLORS
        self.height = 0.4*inch
        
    def draw(self):
        # Draw the continuation header
        header_text = f"{self.title} - Continued..."
        p = Paragraph(header_text, self.section_header_style)
        
        # Create a table for the header
        header_table = Table([[p]], colWidths=[6.5*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.header_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 1, self.COLORS['border']),
        ]))
        
        # Draw the table
        header_table.wrapOn(self.canv, 6.5*inch, self.height)
        header_table.drawOn(self.canv, 0, 0)

def create_pdf_from_row(row_data, row_number):
    """Create a professionally formatted PDF document with containerized sections."""
    buffer = io.BytesIO()
    
    # Get date values for header
    year = None
    month = None
    try:
        if len(row_data) > 0:
            year_val = row_data.iloc[0]
            if not pd.isna(year_val):
                year = str(year_val).strip()
        if len(row_data) > 1:
            month_val = row_data.iloc[1]
            if not pd.isna(month_val):
                month = str(month_val).strip()
    except:
        pass
    
    # Use custom document template
    doc = APGWikiDocTemplate(
        buffer,
        row_number=row_number,
        month=month,
        year=year,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.75*inch,  # Increased for header
        bottomMargin=0.75*inch,  # Increased for footer
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define sophisticated color palette
    COLORS = {
        'primary_dark': colors.HexColor('#1e293b'),     # Slate 800 - Main headers
        'primary': colors.HexColor('#334155'),          # Slate 700 - Section headers
        'secondary': colors.HexColor('#0891b2'),        # Cyan 600 - Accents
        'accent': colors.HexColor('#0e7490'),           # Cyan 700 - Important items
        'success': colors.HexColor('#059669'),          # Emerald 600 - Approvals
        'warning': colors.HexColor('#d97706'),          # Amber 600 - Warnings
        'text_primary': colors.HexColor('#1f2937'),     # Gray 800 - Main text
        'text_secondary': colors.HexColor('#6b7280'),   # Gray 500 - Secondary text
        'border': colors.HexColor('#e5e7eb'),           # Gray 200 - Borders
        'bg_light': colors.HexColor('#f9fafb'),         # Gray 50 - Light backgrounds
        'bg_section': colors.HexColor('#f3f4f6'),       # Gray 100 - Section backgrounds
        'container_1': colors.HexColor('#eff6ff'),      # Blue 50 - Standards container
        'container_2': colors.HexColor('#f0fdfa'),      # Cyan 50 - Issue container
        'container_3': colors.HexColor('#fef3c7'),      # Amber 100 - Technical container
        'container_4': colors.HexColor('#ecfdf5'),      # Emerald 50 - Approval container
        'container_5': colors.HexColor('#fdf4ff'),      # Purple 50 - Documentation container
    }
    
    # Define comprehensive styles
    styles = getSampleStyleSheet()
    
    # Section header style for container titles - reduced size
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    # Field styles for table cells - reduced sizes
    field_label_style = ParagraphStyle(
        'FieldLabel',
        parent=styles['Normal'],
        fontSize=7,
        textColor=COLORS['text_secondary'],
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    field_value_style = ParagraphStyle(
        'FieldValue',
        parent=styles['Normal'],
        fontSize=8,
        textColor=COLORS['text_primary'],
        fontName='Helvetica',
        alignment=TA_LEFT,
        leading=10
    )
    
    # Special styles
    link_style = ParagraphStyle(
        'LinkStyle',
        parent=styles['Normal'],
        textColor=COLORS['secondary'],
        fontSize=8,
        fontName='Courier'
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
    
    # Helper function to create a containerized section with split handling
    def create_container_with_split(title, content_table, bg_color, header_color):
        """Creates a containerized section that can split across pages."""
        # Create header row
        header = Table(
            [[Paragraph(title, section_header_style)]],
            colWidths=[6.5*inch]
        )
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), header_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        # Apply styling to content table with background
        content_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        # Return list of flowables with continuation support
        return [
            KeepTogether([header, content_table]),
            ContinuationTitle(title, section_header_style, header_color, COLORS)
        ]
    
    # Helper function to create a simple container (original style)
    def create_container(title, content_table, bg_color, header_color):
        """Creates a containerized section with header and content."""
        # Create header row
        header = Table(
            [[Paragraph(title, section_header_style)]],
            colWidths=[6.5*inch]
        )
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), header_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        # Combine header and content
        container_data = [[header], [content_table]]
        container = Table(container_data, colWidths=[6.5*inch])
        container.setStyle(TableStyle([
            ('BACKGROUND', (0, 1), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            ('LEFTPADDING', (0, 1), (-1, -1), 10),
            ('RIGHTPADDING', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, 0), 0),
            ('RIGHTPADDING', (0, 0), (-1, 0), 0),
            ('TOPPADDING', (0, 0), (-1, 0), 0),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 0),
        ]))
        return container
    
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
    
    # PAGE 1: Four containers with smaller spacing
    
    # SECTION 1: STANDARDS & PRODUCTS (Container 1 - Blue theme)
    if ifrs_standard or us_gaap or other_standards or related_product or related_platform:
        standards_data = []
        
        # Row 1: IFRS and US GAAP
        row = []
        if ifrs_standard:
            row.append([Paragraph("IFRS Standard", field_label_style),
                       Paragraph(format_value(ifrs_standard), field_value_style)])
        if us_gaap:
            row.append([Paragraph("US GAAP", field_label_style),
                       Paragraph(format_value(us_gaap), field_value_style)])
        if row:
            standards_data.append(row)
        
        # Row 2: Other standards and FI subtopic
        row = []
        if other_standards:
            row.append([Paragraph("Other Standards", field_label_style),
                       Paragraph(format_value(other_standards), field_value_style)])
        if fi_subtopic:
            row.append([Paragraph("FI Subtopic", field_label_style),
                       Paragraph(format_value(fi_subtopic), field_value_style)])
        if row:
            standards_data.append(row)
        
        # Row 3: Product and Platform
        row = []
        if related_product:
            row.append([Paragraph("Related Product", field_label_style),
                       Paragraph(format_value(related_product), field_value_style)])
        if related_platform:
            row.append([Paragraph("Related Platform", field_label_style),
                       Paragraph(format_value(related_platform), field_value_style)])
        if row:
            standards_data.append(row)
        
        # Create the standards table with proper columns
        if standards_data:
            # Flatten the nested structure for the table
            table_data = []
            for row in standards_data:
                if len(row) == 1:
                    # Single item in row - span both columns
                    table_data.append([row[0][0], row[0][1], '', ''])
                elif len(row) == 2:
                    # Two items in row
                    table_data.append([row[0][0], row[0][1], row[1][0], row[1][1]])
            
            standards_table = Table(table_data, colWidths=[1.1*inch, 2.1*inch, 1.1*inch, 2.1*inch])
            standards_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            
            container = create_container(
                "Standards & Products",
                standards_table,
                COLORS['container_1'],
                COLORS['primary']
            )
            story.append(container)
            story.append(Spacer(1, 0.08*inch))
    
    # SECTION 2: TECHNICAL DETAILS & REFERENCES (Container 2 - Amber theme)
    if guidance_ref or differences or benchmarking:
        technical_data = []
        
        if guidance_ref:
            technical_data.append([
                Paragraph("Guidance References", field_label_style),
                Paragraph(format_value(guidance_ref), field_value_style)
            ])
        
        if differences:
            technical_data.append([
                Paragraph("IFRS/US GAAP Differences", field_label_style),
                Paragraph(format_value(differences), field_value_style)
            ])
        
        if benchmarking:
            technical_data.append([
                Paragraph("Benchmarking", field_label_style),
                Paragraph(format_value(benchmarking), field_value_style)
            ])
        
        if technical_data:
            technical_table = Table(technical_data, colWidths=[1.5*inch, 4.8*inch])
            technical_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, COLORS['border']),
            ]))
            
            container = create_container(
                "Technical Details & References",
                technical_table,
                COLORS['container_3'],
                COLORS['warning']
            )
            story.append(container)
            story.append(Spacer(1, 0.08*inch))
    
    # SECTION 3: REVIEW & APPROVALS (Container 3 - Green theme)
    if preparer or stakeholder_concurrence or pwc_concurrence or apg_reviewer:
        approval_data = []
        
        # Create two-column layout for approvals
        row1 = []
        row2 = []
        
        if preparer:
            row1.append([Paragraph("Preparer", field_label_style),
                        Paragraph(format_value(preparer), field_value_style)])
        
        if apg_reviewer:
            row1.append([Paragraph("APG Senior Director", field_label_style),
                        Paragraph(format_value(apg_reviewer), field_value_style)])
        
        if stakeholder_concurrence:
            status = "✓ Yes" if stakeholder_concurrence.lower() in ['yes', 'y', 'true', '1'] else format_value(stakeholder_concurrence)
            row2.append([Paragraph("Stakeholder Concurrence", field_label_style),
                        Paragraph(status, field_value_style)])
        
        if pwc_concurrence:
            status = "✓ Yes" if pwc_concurrence.lower() in ['yes', 'y', 'true', '1'] else format_value(pwc_concurrence)
            row2.append([Paragraph("PwC Concurrence", field_label_style),
                        Paragraph(status, field_value_style)])
        
        # Build the table data
        table_data = []
        if row1:
            if len(row1) == 1:
                table_data.append([row1[0][0], row1[0][1], '', ''])
            elif len(row1) == 2:
                table_data.append([row1[0][0], row1[0][1], row1[1][0], row1[1][1]])
        if row2:
            if len(row2) == 1:
                table_data.append([row2[0][0], row2[0][1], '', ''])
            elif len(row2) == 2:
                table_data.append([row2[0][0], row2[0][1], row2[1][0], row2[1][1]])
        
        if table_data:
            approval_table = Table(table_data, colWidths=[1.4*inch, 1.9*inch, 1.4*inch, 1.9*inch])
            approval_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            
            container = create_container(
                "Review & Approvals",
                approval_table,
                COLORS['container_4'],
                COLORS['success']
            )
            story.append(container)
            story.append(Spacer(1, 0.08*inch))
    
    # SECTION 4: DOCUMENTATION (Container 4 - Purple theme)
    if server_link or key_files or capm_required or capm_date or related_capm:
        doc_data = []
        
        if server_link:
            if server_link.startswith(('http://', 'https://', '\\\\', '//')):
                link_text = f'<link href="{format_value(server_link)}" color="blue">{format_value(server_link)}</link>'
                doc_data.append([
                    Paragraph("Server Link", field_label_style),
                    Paragraph(link_text, link_style)
                ])
            else:
                doc_data.append([
                    Paragraph("Server Link", field_label_style),
                    Paragraph(format_value(server_link), field_value_style)
                ])
        
        if key_files:
            doc_data.append([
                Paragraph("Key Files", field_label_style),
                Paragraph(format_value(key_files), field_value_style)
            ])
        
        if capm_required:
            doc_data.append([
                Paragraph("CAPM Update", field_label_style),
                Paragraph(format_value(capm_required), field_value_style)
            ])
        
        if capm_date:
            doc_data.append([
                Paragraph("CAPM Publication Date", field_label_style),
                Paragraph(format_value(capm_date), field_value_style)
            ])
        
        if related_capm:
            doc_data.append([
                Paragraph("Related CAPM", field_label_style),
                Paragraph(format_value(related_capm), field_value_style)
            ])
        
        if doc_data:
            doc_table = Table(doc_data, colWidths=[1.5*inch, 4.8*inch])
            doc_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, COLORS['border']),
            ]))
            
            container = create_container(
                "Documentation",
                doc_table,
                COLORS['container_5'],
                COLORS['primary']
            )
            story.append(container)
    
    # PAGE BREAK - Move to page 2 for Core Issue Analysis
    story.append(PageBreak())
    
    # PAGE 2: CORE ISSUE ANALYSIS (Container 5 - Cyan theme)
    if accounting_question or conclusion or key_facts:
        issue_data = []
        
        if accounting_question:
            issue_data.append([
                Paragraph("Accounting Question", field_label_style),
                Paragraph(format_value(accounting_question), field_value_style)
            ])
        
        if key_facts:
            issue_data.append([
                Paragraph("Key Facts & Circumstances", field_label_style),
                Paragraph(format_value(key_facts), field_value_style)
            ])
        
        if conclusion:
            issue_data.append([
                Paragraph("Conclusion Reached", field_label_style),
                Paragraph(format_value(conclusion), field_value_style)
            ])
        
        if issue_data:
            # For large content that might split, use the split-aware version
            issue_table = Table(issue_data, colWidths=[1.5*inch, 4.8*inch])
            issue_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, COLORS['border']),
            ]))
            
            # Use the regular container for Core Issue Analysis
            # since it's on its own page and less likely to split
            container = create_container(
                "Core Issue Analysis",
                issue_table,
                COLORS['container_2'],
                COLORS['secondary']
            )
            story.append(container)
    
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