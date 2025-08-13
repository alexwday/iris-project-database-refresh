#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter Assignment Tool for EY Document Processing

This tool allows manual assignment of chapter boundaries to processed PDF pages.
It loads the JSON output from pdf_to_markdown_prep.py and allows users to:
1. View the PDF alongside the JSON data
2. Define chapter start/end pages
3. Name chapters
4. Save the updated JSON with chapter assignments
"""

import sys
import json
import os
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tempfile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QSpinBox, QLineEdit, QGroupBox, QSplitter, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMenuBar, QMenu,
    QStatusBar, QToolBar, QTabWidget, QCheckBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QAction, QIcon, QColor, QFont, QKeySequence

# Try to import PyMuPDF for PDF rendering (optional)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available. External PDF viewer will be used.")

# Import SMB for NAS access (from the prep script)
try:
    from smb.SMBConnection import SMBConnection
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False
    print("pysmb not available. Only local file access will work.")


@dataclass
class ChapterDefinition:
    """Data class for chapter definitions"""
    chapter_number: int
    start_page: int
    end_page: int
    chapter_name: str
    
    def validate(self, total_pages: int) -> List[str]:
        """Validate this chapter definition"""
        errors = []
        if self.start_page < 1:
            errors.append(f"Chapter {self.chapter_number}: Start page must be >= 1")
        if self.end_page > total_pages:
            errors.append(f"Chapter {self.chapter_number}: End page exceeds total pages ({total_pages})")
        if self.start_page > self.end_page:
            errors.append(f"Chapter {self.chapter_number}: Start page > end page")
        if not self.chapter_name.strip():
            errors.append(f"Chapter {self.chapter_number}: Name is empty")
        return errors


class ChapterEditDialog(QDialog):
    """Dialog for editing a single chapter definition"""
    
    def __init__(self, chapter: Optional[ChapterDefinition] = None, 
                 max_pages: int = 9999, existing_chapters: List[ChapterDefinition] = None,
                 parent=None):
        super().__init__(parent)
        self.chapter = chapter
        self.max_pages = max_pages
        self.existing_chapters = existing_chapters or []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Edit Chapter" if self.chapter else "Add Chapter")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Chapter number
        self.number_spin = QSpinBox()
        self.number_spin.setMinimum(1)
        self.number_spin.setMaximum(999)
        if self.chapter:
            self.number_spin.setValue(self.chapter.chapter_number)
        else:
            # Auto-increment from existing chapters
            if self.existing_chapters:
                max_num = max(ch.chapter_number for ch in self.existing_chapters)
                self.number_spin.setValue(max_num + 1)
            else:
                self.number_spin.setValue(1)
        layout.addRow("Chapter Number:", self.number_spin)
        
        # Chapter name
        self.name_input = QLineEdit()
        if self.chapter:
            self.name_input.setText(self.chapter.chapter_name)
        layout.addRow("Chapter Name:", self.name_input)
        
        # Start page
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(1)
        self.start_spin.setMaximum(self.max_pages)
        if self.chapter:
            self.start_spin.setValue(self.chapter.start_page)
        else:
            # Auto-suggest start page (end of last chapter + 1)
            if self.existing_chapters:
                last_end = max(ch.end_page for ch in self.existing_chapters)
                self.start_spin.setValue(last_end + 1)
        layout.addRow("Start Page:", self.start_spin)
        
        # End page
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(1)
        self.end_spin.setMaximum(self.max_pages)
        if self.chapter:
            self.end_spin.setValue(self.chapter.end_page)
        else:
            self.end_spin.setValue(self.start_spin.value() + 10)
        layout.addRow("End Page:", self.end_spin)
        
        # Page count display
        self.page_count_label = QLabel()
        self.update_page_count()
        layout.addRow("Total Pages:", self.page_count_label)
        
        # Connect signals
        self.start_spin.valueChanged.connect(self.update_page_count)
        self.end_spin.valueChanged.connect(self.update_page_count)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
        
    def update_page_count(self):
        """Update the page count display"""
        count = self.end_spin.value() - self.start_spin.value() + 1
        self.page_count_label.setText(f"{count} pages")
        
    def get_chapter(self) -> ChapterDefinition:
        """Get the chapter definition from the dialog"""
        return ChapterDefinition(
            chapter_number=self.number_spin.value(),
            start_page=self.start_spin.value(),
            end_page=self.end_spin.value(),
            chapter_name=self.name_input.text().strip()
        )


class ChapterAssignmentTool(QMainWindow):
    """Main window for the chapter assignment tool"""
    
    def __init__(self):
        super().__init__()
        self.json_data = []
        self.json_file_path = None
        self.pdf_file_path = None
        self.chapters = []
        self.current_page = 1
        self.total_pages = 0
        self.pdf_doc = None  # For PyMuPDF if available
        self.unsaved_changes = False
        
        self.init_ui()
        self.create_menus()
        self.update_ui_state()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Chapter Assignment Tool - EY Document Processing")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # File info bar
        self.info_label = QLabel("No file loaded")
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        main_layout.addWidget(self.info_label)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Chapter definitions
        chapter_widget = self.create_chapter_panel()
        splitter.addWidget(chapter_widget)
        
        # Right panel: Viewer and navigation
        viewer_widget = self.create_viewer_panel()
        splitter.addWidget(viewer_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Open JSON action
        open_action = QAction("Open JSON", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_json_file)
        toolbar.addAction(open_action)
        
        # Save action
        save_action = QAction("Save Changes", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_to_json)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Open PDF action
        open_pdf_action = QAction("Open PDF", self)
        open_pdf_action.triggered.connect(self.open_pdf_file)
        toolbar.addAction(open_pdf_action)
        
        # Open PDF in external viewer
        external_pdf_action = QAction("Open PDF Externally", self)
        external_pdf_action.triggered.connect(self.open_pdf_external)
        toolbar.addAction(external_pdf_action)
        
        toolbar.addSeparator()
        
        # Validate action
        validate_action = QAction("Validate Chapters", self)
        validate_action.triggered.connect(self.validate_chapters)
        toolbar.addAction(validate_action)
        
    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open JSON...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_json_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_to_json)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_as_json)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        add_chapter_action = QAction("Add Chapter", self)
        add_chapter_action.setShortcut("Ctrl+N")
        add_chapter_action.triggered.connect(self.add_chapter)
        edit_menu.addAction(add_chapter_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        validate_action = QAction("Validate Chapters", self)
        validate_action.triggered.connect(self.validate_chapters)
        tools_menu.addAction(validate_action)
        
        auto_detect_action = QAction("Auto-Detect Chapters", self)
        auto_detect_action.triggered.connect(self.auto_detect_chapters)
        tools_menu.addAction(auto_detect_action)
        
    def create_chapter_panel(self) -> QWidget:
        """Create the chapter definition panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Chapter list group
        group = QGroupBox("Chapter Definitions")
        group_layout = QVBoxLayout()
        
        # Add chapter button
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Chapter")
        add_btn.clicked.connect(self.add_chapter)
        btn_layout.addWidget(add_btn)
        
        auto_detect_btn = QPushButton("Auto-Detect")
        auto_detect_btn.clicked.connect(self.auto_detect_chapters)
        btn_layout.addWidget(auto_detect_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_chapters)
        btn_layout.addWidget(clear_btn)
        
        group_layout.addLayout(btn_layout)
        
        # Chapter list table
        self.chapter_table = QTableWidget()
        self.chapter_table.setColumnCount(5)
        self.chapter_table.setHorizontalHeaderLabels(
            ["#", "Name", "Start", "End", "Pages"]
        )
        self.chapter_table.horizontalHeader().setStretchLastSection(True)
        self.chapter_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.chapter_table.itemDoubleClicked.connect(self.edit_chapter_at_row)
        group_layout.addWidget(self.chapter_table)
        
        # Chapter actions
        action_layout = QHBoxLayout()
        edit_btn = QPushButton("Edit Selected")
        edit_btn.clicked.connect(self.edit_selected_chapter)
        action_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_chapter)
        action_layout.addWidget(delete_btn)
        
        group_layout.addLayout(action_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Quick add section
        quick_group = QGroupBox("Quick Add Chapter")
        quick_layout = QHBoxLayout()
        
        quick_layout.addWidget(QLabel("Start:"))
        self.quick_start = QSpinBox()
        self.quick_start.setMinimum(1)
        self.quick_start.setMaximum(99999)
        quick_layout.addWidget(self.quick_start)
        
        quick_layout.addWidget(QLabel("End:"))
        self.quick_end = QSpinBox()
        self.quick_end.setMinimum(1)
        self.quick_end.setMaximum(99999)
        quick_layout.addWidget(self.quick_end)
        
        quick_layout.addWidget(QLabel("Name:"))
        self.quick_name = QLineEdit()
        self.quick_name.setPlaceholderText("Chapter name...")
        quick_layout.addWidget(self.quick_name)
        
        quick_add_btn = QPushButton("Add")
        quick_add_btn.clicked.connect(self.quick_add_chapter)
        quick_layout.addWidget(quick_add_btn)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        return widget
        
    def create_viewer_panel(self) -> QWidget:
        """Create the PDF viewer panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        
        # Page navigation buttons
        first_btn = QPushButton("<<")
        first_btn.clicked.connect(lambda: self.go_to_page(1))
        nav_layout.addWidget(first_btn)
        
        prev_btn = QPushButton("<")
        prev_btn.clicked.connect(lambda: self.go_to_page(self.current_page - 1))
        nav_layout.addWidget(prev_btn)
        
        nav_layout.addWidget(QLabel("Page:"))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.valueChanged.connect(self.go_to_page)
        nav_layout.addWidget(self.page_spin)
        
        self.page_label = QLabel("/ 0")
        nav_layout.addWidget(self.page_label)
        
        next_btn = QPushButton(">")
        next_btn.clicked.connect(lambda: self.go_to_page(self.current_page + 1))
        nav_layout.addWidget(next_btn)
        
        last_btn = QPushButton(">>")
        last_btn.clicked.connect(lambda: self.go_to_page(self.total_pages))
        nav_layout.addWidget(last_btn)
        
        nav_layout.addStretch()
        
        # Page reference display
        nav_layout.addWidget(QLabel("Page Ref:"))
        self.page_ref_label = QLabel("-")
        self.page_ref_label.setStyleSheet("QLabel { font-weight: bold; }")
        nav_layout.addWidget(self.page_ref_label)
        
        nav_layout.addStretch()
        
        # Open PDF externally button
        open_pdf_btn = QPushButton("Open PDF in Viewer")
        open_pdf_btn.clicked.connect(self.open_pdf_external)
        nav_layout.addWidget(open_pdf_btn)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Content display tabs
        self.content_tabs = QTabWidget()
        
        # Markdown content tab
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        self.content_display.setFont(QFont("Courier", 10))
        self.content_tabs.addTab(self.content_display, "Markdown Content")
        
        # Chapter info tab
        self.chapter_info = QTextEdit()
        self.chapter_info.setReadOnly(True)
        self.content_tabs.addTab(self.chapter_info, "Chapter Info")
        
        # If PyMuPDF is available, add PDF preview tab
        if PYMUPDF_AVAILABLE:
            self.pdf_preview = QLabel()
            self.pdf_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pdf_preview.setStyleSheet("QLabel { background-color: #e0e0e0; }")
            self.content_tabs.addTab(self.pdf_preview, "PDF Preview")
        
        layout.addWidget(self.content_tabs)
        
        return widget
        
    def open_json_file(self):
        """Open a JSON file from disk"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.load_json_file(file_path)
            
    def load_json_file(self, file_path: str):
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
                
            self.json_file_path = file_path
            
            # Extract info from JSON
            if self.json_data:
                self.total_pages = len(self.json_data)
                first_record = self.json_data[0]
                filename = first_record.get('filename', 'Unknown')
                
                # Update UI
                self.info_label.setText(
                    f"File: {os.path.basename(file_path)} | "
                    f"PDF: {filename} | "
                    f"Total Pages: {self.total_pages}"
                )
                
                # Update page spinner
                self.page_spin.setMaximum(self.total_pages)
                self.quick_start.setMaximum(self.total_pages)
                self.quick_end.setMaximum(self.total_pages)
                
                # Load existing chapters if any
                self.load_existing_chapters()
                
                # Go to first page
                self.go_to_page(1)
                
                self.status_bar.showMessage(f"Loaded {self.total_pages} pages")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON file:\n{str(e)}")
            
    def load_existing_chapters(self):
        """Check if JSON already has chapter assignments and load them"""
        self.chapters.clear()
        
        # Check if first record has chapter fields
        if self.json_data and 'chapter_number' in self.json_data[0]:
            # Extract unique chapters
            chapters_dict = {}
            for record in self.json_data:
                if 'chapter_number' in record and record['chapter_number']:
                    ch_num = record['chapter_number']
                    page_num = record['page_number']
                    
                    if ch_num not in chapters_dict:
                        chapters_dict[ch_num] = {
                            'number': ch_num,
                            'name': record.get('chapter_name', f'Chapter {ch_num}'),
                            'start': page_num,
                            'end': page_num
                        }
                    else:
                        chapters_dict[ch_num]['end'] = max(
                            chapters_dict[ch_num]['end'], page_num
                        )
                        
            # Convert to ChapterDefinition objects
            for ch_data in sorted(chapters_dict.values(), key=lambda x: x['number']):
                chapter = ChapterDefinition(
                    chapter_number=ch_data['number'],
                    start_page=ch_data['start'],
                    end_page=ch_data['end'],
                    chapter_name=ch_data['name']
                )
                self.chapters.append(chapter)
                
            self.update_chapter_table()
            self.status_bar.showMessage(f"Loaded {len(self.chapters)} existing chapters")
            
    def open_pdf_file(self):
        """Open a PDF file for preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_file_path = file_path
            if PYMUPDF_AVAILABLE:
                try:
                    self.pdf_doc = fitz.open(file_path)
                    self.status_bar.showMessage(f"Opened PDF: {os.path.basename(file_path)}")
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to open PDF:\n{str(e)}")
                    
    def open_pdf_external(self):
        """Open PDF in external viewer"""
        if not self.pdf_file_path:
            # Try to get PDF path from JSON data
            if self.json_data:
                # Prompt for PDF file
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select PDF File", "", "PDF Files (*.pdf)"
                )
                if file_path:
                    self.pdf_file_path = file_path
                    
        if self.pdf_file_path and os.path.exists(self.pdf_file_path):
            try:
                if platform.system() == 'Darwin':  # macOS
                    subprocess.Popen(['open', self.pdf_file_path])
                elif platform.system() == 'Windows':
                    os.startfile(self.pdf_file_path)
                else:  # Linux
                    subprocess.Popen(['xdg-open', self.pdf_file_path])
                self.status_bar.showMessage("Opened PDF in external viewer")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to open PDF:\n{str(e)}")
        else:
            QMessageBox.information(self, "No PDF", "Please select a PDF file first")
            
    def go_to_page(self, page_num: int):
        """Navigate to a specific page"""
        if not self.json_data:
            return
            
        page_num = max(1, min(page_num, self.total_pages))
        self.current_page = page_num
        
        # Update spinner
        self.page_spin.setValue(page_num)
        
        # Get page data
        page_data = self.json_data[page_num - 1]
        
        # Update content display
        content = page_data.get('content', '')
        self.content_display.setPlainText(content)
        
        # Update page reference
        page_ref = page_data.get('page_reference', '-')
        self.page_ref_label.setText(str(page_ref) if page_ref else '-')
        
        # Update chapter info
        self.update_chapter_info_display()
        
        # Update PDF preview if available
        if PYMUPDF_AVAILABLE and self.pdf_doc:
            self.update_pdf_preview()
            
    def update_chapter_info_display(self):
        """Update the chapter info display for current page"""
        info_text = f"Page {self.current_page} of {self.total_pages}\n\n"
        
        # Find which chapter this page belongs to
        for chapter in self.chapters:
            if chapter.start_page <= self.current_page <= chapter.end_page:
                info_text += f"Current Chapter: {chapter.chapter_number}\n"
                info_text += f"Chapter Name: {chapter.chapter_name}\n"
                info_text += f"Chapter Range: {chapter.start_page}-{chapter.end_page}\n"
                info_text += f"Position in Chapter: Page {self.current_page - chapter.start_page + 1} "
                info_text += f"of {chapter.end_page - chapter.start_page + 1}"
                break
        else:
            info_text += "Not assigned to any chapter"
            
        self.chapter_info.setPlainText(info_text)
        
    def update_pdf_preview(self):
        """Update PDF preview if PyMuPDF is available"""
        # This is a placeholder - actual PDF rendering would go here
        if self.pdf_doc and self.current_page <= len(self.pdf_doc):
            self.pdf_preview.setText(f"PDF Preview for page {self.current_page}\n(Preview not implemented)")
            
    def add_chapter(self):
        """Add a new chapter definition"""
        dialog = ChapterEditDialog(
            chapter=None,
            max_pages=self.total_pages,
            existing_chapters=self.chapters,
            parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_chapter = dialog.get_chapter()
            self.chapters.append(new_chapter)
            self.chapters.sort(key=lambda x: x.start_page)
            self.update_chapter_table()
            self.mark_unsaved_changes()
            
    def quick_add_chapter(self):
        """Quick add a chapter from the quick add fields"""
        if not self.quick_name.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a chapter name")
            return
            
        # Auto-increment chapter number
        if self.chapters:
            next_num = max(ch.chapter_number for ch in self.chapters) + 1
        else:
            next_num = 1
            
        new_chapter = ChapterDefinition(
            chapter_number=next_num,
            start_page=self.quick_start.value(),
            end_page=self.quick_end.value(),
            chapter_name=self.quick_name.text().strip()
        )
        
        # Validate
        errors = new_chapter.validate(self.total_pages)
        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(errors))
            return
            
        self.chapters.append(new_chapter)
        self.chapters.sort(key=lambda x: x.start_page)
        self.update_chapter_table()
        self.mark_unsaved_changes()
        
        # Clear quick add fields
        self.quick_name.clear()
        if self.chapters:
            self.quick_start.setValue(self.chapters[-1].end_page + 1)
            
    def edit_selected_chapter(self):
        """Edit the selected chapter"""
        row = self.chapter_table.currentRow()
        if row >= 0:
            self.edit_chapter_at_row(None)
            
    def edit_chapter_at_row(self, item):
        """Edit chapter at specific row"""
        row = self.chapter_table.currentRow()
        if row >= 0 and row < len(self.chapters):
            chapter = self.chapters[row]
            dialog = ChapterEditDialog(
                chapter=chapter,
                max_pages=self.total_pages,
                existing_chapters=[ch for ch in self.chapters if ch != chapter],
                parent=self
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.chapters[row] = dialog.get_chapter()
                self.chapters.sort(key=lambda x: x.start_page)
                self.update_chapter_table()
                self.mark_unsaved_changes()
                
    def delete_selected_chapter(self):
        """Delete the selected chapter"""
        row = self.chapter_table.currentRow()
        if row >= 0 and row < len(self.chapters):
            chapter = self.chapters[row]
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete Chapter {chapter.chapter_number}: {chapter.chapter_name}?"
            )
            if reply == QMessageBox.StandardButton.Yes:
                del self.chapters[row]
                self.update_chapter_table()
                self.mark_unsaved_changes()
                
    def clear_all_chapters(self):
        """Clear all chapter definitions"""
        if self.chapters:
            reply = QMessageBox.question(
                self, "Confirm Clear",
                f"Clear all {len(self.chapters)} chapter definitions?"
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.chapters.clear()
                self.update_chapter_table()
                self.mark_unsaved_changes()
                
    def update_chapter_table(self):
        """Update the chapter table display"""
        self.chapter_table.setRowCount(len(self.chapters))
        
        for row, chapter in enumerate(self.chapters):
            # Chapter number
            item = QTableWidgetItem(str(chapter.chapter_number))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chapter_table.setItem(row, 0, item)
            
            # Chapter name
            self.chapter_table.setItem(row, 1, QTableWidgetItem(chapter.chapter_name))
            
            # Start page
            item = QTableWidgetItem(str(chapter.start_page))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chapter_table.setItem(row, 2, item)
            
            # End page
            item = QTableWidgetItem(str(chapter.end_page))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chapter_table.setItem(row, 3, item)
            
            # Page count
            page_count = chapter.end_page - chapter.start_page + 1
            item = QTableWidgetItem(str(page_count))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chapter_table.setItem(row, 4, item)
            
            # Highlight any issues
            errors = chapter.validate(self.total_pages)
            if errors:
                for col in range(5):
                    self.chapter_table.item(row, col).setBackground(QColor(255, 200, 200))
                    
    def validate_chapters(self):
        """Validate all chapter definitions"""
        if not self.chapters:
            QMessageBox.information(self, "Validation", "No chapters defined")
            return
            
        errors = []
        warnings = []
        
        # Individual chapter validation
        for chapter in self.chapters:
            chapter_errors = chapter.validate(self.total_pages)
            errors.extend(chapter_errors)
            
        # Check for overlaps
        for i, ch1 in enumerate(self.chapters):
            for ch2 in self.chapters[i+1:]:
                if ch1.end_page >= ch2.start_page and ch1.start_page <= ch2.end_page:
                    errors.append(
                        f"Chapters {ch1.chapter_number} and {ch2.chapter_number} overlap"
                    )
                    
        # Check for gaps
        sorted_chapters = sorted(self.chapters, key=lambda x: x.start_page)
        for i in range(len(sorted_chapters) - 1):
            if sorted_chapters[i].end_page + 1 < sorted_chapters[i+1].start_page:
                gap_start = sorted_chapters[i].end_page + 1
                gap_end = sorted_chapters[i+1].start_page - 1
                warnings.append(
                    f"Gap between chapters {sorted_chapters[i].chapter_number} and "
                    f"{sorted_chapters[i+1].chapter_number}: pages {gap_start}-{gap_end}"
                )
                
        # Check coverage
        if sorted_chapters:
            if sorted_chapters[0].start_page > 1:
                warnings.append(f"Pages 1-{sorted_chapters[0].start_page - 1} not covered")
            if sorted_chapters[-1].end_page < self.total_pages:
                warnings.append(
                    f"Pages {sorted_chapters[-1].end_page + 1}-{self.total_pages} not covered"
                )
                
        # Display results
        if errors or warnings:
            msg = ""
            if errors:
                msg += "ERRORS:\n" + "\n".join(errors) + "\n\n"
            if warnings:
                msg += "WARNINGS:\n" + "\n".join(warnings)
            QMessageBox.warning(self, "Validation Results", msg)
        else:
            QMessageBox.information(self, "Validation Results", "All chapters valid!")
            
    def auto_detect_chapters(self):
        """Attempt to auto-detect chapters based on content patterns"""
        if not self.json_data:
            QMessageBox.warning(self, "Warning", "Please load a JSON file first")
            return
            
        reply = QMessageBox.question(
            self, "Auto-Detect Chapters",
            "This will clear existing chapters and attempt to detect new ones.\nContinue?"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        self.chapters.clear()
        
        # Simple heuristic: look for pages with "Chapter" in the content
        chapter_pages = []
        for i, page_data in enumerate(self.json_data):
            content = page_data.get('content', '')
            # Look for chapter markers
            if any(marker in content[:500] for marker in ['# Chapter', '## Chapter', 'CHAPTER']):
                chapter_pages.append(i + 1)
                
        if not chapter_pages:
            QMessageBox.information(self, "Auto-Detect", "No chapters detected")
            return
            
        # Create chapters from detected pages
        for i, start_page in enumerate(chapter_pages):
            if i < len(chapter_pages) - 1:
                end_page = chapter_pages[i + 1] - 1
            else:
                end_page = self.total_pages
                
            chapter = ChapterDefinition(
                chapter_number=i + 1,
                start_page=start_page,
                end_page=end_page,
                chapter_name=f"Chapter {i + 1}"
            )
            self.chapters.append(chapter)
            
        self.update_chapter_table()
        self.mark_unsaved_changes()
        QMessageBox.information(
            self, "Auto-Detect Complete",
            f"Detected {len(self.chapters)} chapters"
        )
        
    def save_to_json(self):
        """Save chapter assignments to JSON"""
        if not self.json_file_path:
            self.save_as_json()
            return
            
        self.save_json_to_file(self.json_file_path)
        
    def save_as_json(self):
        """Save chapter assignments to a new JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.save_json_to_file(file_path)
            self.json_file_path = file_path
            
    def save_json_to_file(self, file_path: str):
        """Save the updated JSON data to file"""
        if not self.json_data:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
            
        try:
            # Apply chapter assignments to JSON data
            for record in self.json_data:
                page_num = record['page_number']
                
                # Find which chapter this page belongs to
                chapter_assigned = False
                for chapter in self.chapters:
                    if chapter.start_page <= page_num <= chapter.end_page:
                        record['chapter_number'] = chapter.chapter_number
                        record['chapter_name'] = chapter.chapter_name
                        chapter_assigned = True
                        break
                        
                if not chapter_assigned:
                    # Clear chapter fields if not in any chapter
                    record.pop('chapter_number', None)
                    record.pop('chapter_name', None)
                    
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
                
            self.unsaved_changes = False
            self.status_bar.showMessage(f"Saved to {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", "Chapter assignments saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
            
    def mark_unsaved_changes(self):
        """Mark that there are unsaved changes"""
        self.unsaved_changes = True
        if self.json_file_path:
            self.setWindowTitle(f"Chapter Assignment Tool - {os.path.basename(self.json_file_path)}*")
        else:
            self.setWindowTitle("Chapter Assignment Tool*")
            
    def update_ui_state(self):
        """Update UI elements based on current state"""
        has_data = bool(self.json_data)
        
        # Update spinners
        if has_data:
            self.page_spin.setMaximum(self.total_pages)
            self.quick_start.setMaximum(self.total_pages)
            self.quick_end.setMaximum(self.total_pages)
            
    def closeEvent(self, event):
        """Handle window close event"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_to_json()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Chapter Assignment Tool")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ChapterAssignmentTool()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()