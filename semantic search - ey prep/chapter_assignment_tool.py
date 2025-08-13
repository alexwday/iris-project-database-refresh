#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter Assignment Tool for EY Document Processing

Streamlined interface focused on markdown content viewing and chapter assignment.
Features include Azure tag handling, fixed header/footer display, and chapter auto-detection.
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QSpinBox, QLineEdit, QGroupBox, 
    QSplitter, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, 
    QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, 
    QMenuBar, QStatusBar, QToolBar, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QTextCursor, QCursor


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
        self.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        # Chapter number (allow 0 for front matter)
        self.number_spin = QSpinBox()
        self.number_spin.setMinimum(0)
        self.number_spin.setMaximum(999)
        if self.chapter:
            self.number_spin.setValue(self.chapter.chapter_number)
        else:
            if self.existing_chapters:
                max_num = max(ch.chapter_number for ch in self.existing_chapters)
                self.number_spin.setValue(max_num + 1)
            else:
                self.number_spin.setValue(0)  # Start with 0 for front matter
        layout.addRow("Chapter Number:", self.number_spin)
        
        # Chapter name
        self.name_input = QLineEdit()
        if self.chapter:
            self.name_input.setText(self.chapter.chapter_name)
        self.name_input.setPlaceholderText("Enter chapter name...")
        layout.addRow("Chapter Name:", self.name_input)
        
        # Start page
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(1)
        self.start_spin.setMaximum(self.max_pages)
        if self.chapter:
            self.start_spin.setValue(self.chapter.start_page)
        else:
            if self.existing_chapters:
                last_end = max(ch.end_page for ch in self.existing_chapters)
                self.start_spin.setValue(min(last_end + 1, self.max_pages))
        layout.addRow("Start Page:", self.start_spin)
        
        # End page
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(1)
        self.end_spin.setMaximum(self.max_pages)
        if self.chapter:
            self.end_spin.setValue(self.chapter.end_page)
        else:
            self.end_spin.setValue(min(self.start_spin.value() + 10, self.max_pages))
        layout.addRow("End Page:", self.end_spin)
        
        # Page count display
        self.page_count_label = QLabel()
        self.page_count_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
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
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
        
    def update_page_count(self):
        """Update the page count display"""
        count = self.end_spin.value() - self.start_spin.value() + 1
        self.page_count_label.setText(f"{count} pages")
        
    def validate_and_accept(self):
        """Validate input before accepting"""
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Chapter name cannot be empty")
            return
        self.accept()
        
    def get_chapter(self) -> ChapterDefinition:
        """Get the chapter definition from the dialog"""
        return ChapterDefinition(
            chapter_number=self.number_spin.value(),
            start_page=self.start_spin.value(),
            end_page=self.end_spin.value(),
            chapter_name=self.name_input.text().strip()
        )


class ChapterAssignmentTool(QMainWindow):
    """Main window for the chapter assignment tool - Streamlined Version"""
    
    def __init__(self):
        super().__init__()
        self.json_data = []
        self.json_file_path = None
        self.chapters = []
        self.current_page = 1
        self.total_pages = 0
        self.unsaved_changes = False
        
        self.init_ui()
        self.create_menus()
        self.update_ui_state()
        
    def init_ui(self):
        """Initialize the streamlined user interface"""
        self.setWindowTitle("Chapter Assignment Tool - Document Processing")
        self.setGeometry(100, 100, 1600, 900)
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 5px 15px;
                border-radius: 3px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTableWidget {
                background-color: white;
                gridline-color: #e0e0e0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11pt;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        
        # Create toolbar
        self.create_toolbar()
        
        # File info bar (compact)
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.Box)
        info_frame.setMaximumHeight(35)  # Fix the height issue
        info_frame.setStyleSheet("QFrame { background-color: #e8f4f8; border: 1px solid #b0d4e3; }")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 5, 10, 5)  # Reduce margins
        
        self.info_label = QLabel("No file loaded")
        self.info_label.setStyleSheet("QLabel { border: none; font-size: 10pt; }")
        info_layout.addWidget(self.info_label)
        
        # Page info on the right
        self.page_info_label = QLabel("")
        self.page_info_label.setStyleSheet("QLabel { border: none; font-size: 10pt; color: #666; }")
        info_layout.addWidget(self.page_info_label)
        
        main_layout.addWidget(info_frame)
        
        # Create main content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Chapter management (narrower)
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel: Content viewer (wider)
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions (30% left, 70% right)
        content_splitter.setSizes([500, 1100])
        content_splitter.setStretchFactor(0, 0)  # Don't stretch left panel
        content_splitter.setStretchFactor(1, 1)  # Stretch right panel
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2c3e50;
                spacing: 3px;
                padding: 5px;
            }
            QToolBar QToolButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 8px;
                margin: 2px;
                border-radius: 3px;
                font-weight: bold;
            }
            QToolBar QToolButton:hover {
                background-color: #4a5f7f;
            }
        """)
        self.addToolBar(toolbar)
        
        # File operations
        open_action = QAction("📁 Open JSON", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_json_file)
        toolbar.addAction(open_action)
        
        save_action = QAction("💾 Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_to_json)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Chapter operations
        add_action = QAction("➕ Add Chapter", self)
        add_action.setShortcut("Ctrl+N")
        add_action.triggered.connect(self.add_chapter)
        toolbar.addAction(add_action)
        
        auto_detect_action = QAction("🔍 Auto-Detect", self)
        auto_detect_action.triggered.connect(self.auto_detect_chapters)
        toolbar.addAction(auto_detect_action)
        
        validate_action = QAction("✓ Validate", self)
        validate_action.triggered.connect(self.validate_chapters)
        toolbar.addAction(validate_action)
        
        toolbar.addSeparator()
        
        # Navigation
        first_action = QAction("⏮ First", self)
        first_action.triggered.connect(lambda: self.go_to_page(1))
        toolbar.addAction(first_action)
        
        prev_action = QAction("◀ Previous", self)
        prev_action.triggered.connect(lambda: self.go_to_page(max(1, self.current_page - 1)))
        toolbar.addAction(prev_action)
        
        next_action = QAction("▶ Next", self)
        next_action.triggered.connect(lambda: self.go_to_page(min(self.total_pages, self.current_page + 1)))
        toolbar.addAction(next_action)
        
        last_action = QAction("⏭ Last", self)
        last_action.triggered.connect(lambda: self.go_to_page(self.total_pages))
        toolbar.addAction(last_action)
        
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
        
        # Navigation menu
        nav_menu = menubar.addMenu("Navigate")
        
        go_to_action = QAction("Go to Page...", self)
        go_to_action.setShortcut("Ctrl+G")
        go_to_action.triggered.connect(self.show_go_to_dialog)
        nav_menu.addAction(go_to_action)
        
    def create_left_panel(self) -> QWidget:
        """Create the left panel with chapter management"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Quick Add Section (at the top for easy access)
        quick_group = QGroupBox("Quick Add Chapter")
        quick_group.setMaximumHeight(180)
        quick_layout = QVBoxLayout()
        
        # Row 1: Chapter number and name
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Ch#:"))
        self.quick_chapter_num = QSpinBox()
        self.quick_chapter_num.setMinimum(0)  # Allow chapter 0
        self.quick_chapter_num.setMaximum(999)
        self.quick_chapter_num.setMaximumWidth(60)
        row1.addWidget(self.quick_chapter_num)
        
        row1.addWidget(QLabel("Name:"))
        self.quick_name = QLineEdit()
        self.quick_name.setPlaceholderText("Chapter name...")
        row1.addWidget(self.quick_name)
        quick_layout.addLayout(row1)
        
        # Row 2: Page range
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pages:"))
        self.quick_start = QSpinBox()
        self.quick_start.setMinimum(1)
        self.quick_start.setMaximum(99999)
        self.quick_start.valueChanged.connect(self.update_quick_page_count)
        row2.addWidget(self.quick_start)
        
        row2.addWidget(QLabel("to"))
        self.quick_end = QSpinBox()
        self.quick_end.setMinimum(1)
        self.quick_end.setMaximum(99999)
        self.quick_end.valueChanged.connect(self.update_quick_page_count)
        row2.addWidget(self.quick_end)
        
        self.quick_page_count = QLabel("(0 pages)")
        self.quick_page_count.setStyleSheet("QLabel { color: #666; }")
        row2.addWidget(self.quick_page_count)
        quick_layout.addLayout(row2)
        
        # Row 3: Buttons
        button_row = QHBoxLayout()
        quick_add_btn = QPushButton("Add Chapter")
        quick_add_btn.clicked.connect(self.quick_add_chapter)
        button_row.addWidget(quick_add_btn)
        
        use_current_btn = QPushButton("Use Current Page")
        use_current_btn.setStyleSheet("QPushButton { background-color: #2196F3; }")
        use_current_btn.clicked.connect(self.use_current_page_for_quick)
        button_row.addWidget(use_current_btn)
        quick_layout.addLayout(button_row)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # Chapter list
        chapters_group = QGroupBox("Chapter Definitions")
        chapters_layout = QVBoxLayout()
        
        # Chapter table with clickable pages
        self.chapter_table = QTableWidget()
        self.chapter_table.setColumnCount(5)
        self.chapter_table.setHorizontalHeaderLabels(
            ["#", "Name", "Start", "End", "Pages"]
        )
        
        # Set smaller font for table
        table_font = QFont()
        table_font.setPointSize(9)
        self.chapter_table.setFont(table_font)
        
        # Make columns interactive
        self.chapter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.chapter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.chapter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.chapter_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.chapter_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        
        self.chapter_table.setColumnWidth(0, 30)
        self.chapter_table.setColumnWidth(2, 50)
        self.chapter_table.setColumnWidth(3, 50)
        self.chapter_table.setColumnWidth(4, 50)
        
        # Enable word wrap and adjust row heights
        self.chapter_table.setWordWrap(True)
        self.chapter_table.verticalHeader().setDefaultSectionSize(40)
        self.chapter_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        self.chapter_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.chapter_table.itemDoubleClicked.connect(self.on_chapter_item_clicked)
        self.chapter_table.itemClicked.connect(self.on_chapter_item_clicked)
        
        chapters_layout.addWidget(self.chapter_table)
        
        # Chapter action buttons
        action_layout = QHBoxLayout()
        
        edit_btn = QPushButton("Edit")
        edit_btn.setStyleSheet("QPushButton { background-color: #FF9800; }")
        edit_btn.clicked.connect(self.edit_selected_chapter)
        action_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        delete_btn.clicked.connect(self.delete_selected_chapter)
        action_layout.addWidget(delete_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("QPushButton { background-color: #9E9E9E; }")
        clear_btn.clicked.connect(self.clear_all_chapters)
        action_layout.addWidget(clear_btn)
        
        chapters_layout.addLayout(action_layout)
        
        chapters_group.setLayout(chapters_layout)
        layout.addWidget(chapters_group)
        
        return widget
        
    def create_right_panel(self) -> QWidget:
        """Create the right panel with full-height content viewer"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        # Navigation bar
        nav_frame = QFrame()
        nav_frame.setFrameStyle(QFrame.Shape.Box)
        nav_frame.setMaximumHeight(50)
        nav_layout = QHBoxLayout(nav_frame)
        
        # Page navigation
        nav_layout.addWidget(QLabel("Page:"))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.setMaximumWidth(80)
        self.page_spin.valueChanged.connect(self.go_to_page)
        nav_layout.addWidget(self.page_spin)
        
        self.page_label = QLabel("/ 0")
        nav_layout.addWidget(self.page_label)
        
        # Page reference
        nav_layout.addWidget(QLabel("  |  Reference:"))
        self.page_ref_label = QLabel("-")
        self.page_ref_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        nav_layout.addWidget(self.page_ref_label)
        
        # Chapter indicator
        self.chapter_indicator = QLabel("")
        self.chapter_indicator.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        nav_layout.addWidget(self.chapter_indicator)
        
        nav_layout.addStretch()
        
        # Search in content
        nav_layout.addWidget(QLabel("Find:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in page...")
        self.search_input.setMaximumWidth(200)
        self.search_input.textChanged.connect(self.search_in_content)
        nav_layout.addWidget(self.search_input)
        
        layout.addWidget(nav_frame)
        
        # Full-height markdown content viewer with fixed header/footer
        content_group = QGroupBox("Markdown Content")
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)
        
        # Fixed header row for PageHeader
        self.page_header_label = QLabel("")
        self.page_header_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                border: 1px solid #ccc; 
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                color: #333;
            }
        """)
        self.page_header_label.setWordWrap(True)
        self.page_header_label.setMaximumHeight(40)
        content_layout.addWidget(self.page_header_label)
        
        # Main content display
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        self.content_display.setFont(QFont("Consolas", 11))
        self.content_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        content_layout.addWidget(self.content_display)
        
        # Fixed footer row for PageFooter
        self.page_footer_label = QLabel("")
        self.page_footer_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                border: 1px solid #ccc; 
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                color: #333;
            }
        """)
        self.page_footer_label.setWordWrap(True)
        self.page_footer_label.setMaximumHeight(40)
        content_layout.addWidget(self.page_footer_label)
        
        content_group.setLayout(content_layout)
        layout.addWidget(content_group)
        
        return widget
        
    def update_quick_page_count(self):
        """Update the page count in quick add section"""
        count = self.quick_end.value() - self.quick_start.value() + 1
        self.quick_page_count.setText(f"({count} pages)")
        
    def use_current_page_for_quick(self):
        """Use current page as start for quick add"""
        if self.json_data:
            self.quick_start.setValue(self.current_page)
            self.quick_end.setValue(min(self.current_page + 20, self.total_pages))
            # Try to extract chapter name from current page
            content = self.json_data[self.current_page - 1].get('content', '')
            name = self.extract_chapter_name(content)
            if name:
                self.quick_name.setText(name)
            self.quick_name.setFocus()
            
    def extract_chapter_name(self, content: str) -> str:
        """Extract a chapter name from content, skipping Azure tags"""
        lines = content.split('\n')[:20]  # Check more lines to skip past tags
        for line in lines:
            line = line.strip()
            
            # Skip Azure comment tags
            if line.startswith('<!--') and line.endswith('-->'):
                continue
            
            # Look for markdown headers (# or ## or ###)
            if line.startswith('#'):
                # Remove the # symbols and clean up
                line = re.sub(r'^#+\s*', '', line)
                # Remove "Chapter X:" patterns but keep the rest
                line = re.sub(r'^Chapter\s+\d+[:\s]*', '', line, flags=re.IGNORECASE)
                # Remove just numbers at the start
                line = re.sub(r'^\d+\.?\s*', '', line)
                # If we have something substantial, use it
                if len(line) > 3:
                    return line[:100]  # Limit length to 100 chars
                    
        return ""
        
    def on_chapter_item_clicked(self, item):
        """Handle clicking on chapter table items"""
        if item and item.column() in [2, 3]:  # Start or End column
            try:
                page_num = int(item.text())
                self.go_to_page(page_num)
                # Highlight the relevant content
                self.status_bar.showMessage(f"Jumped to page {page_num}")
            except ValueError:
                pass
                
    def search_in_content(self, text):
        """Search for text in the current content display"""
        if not text:
            # Clear highlighting
            cursor = self.content_display.textCursor()
            cursor.clearSelection()
            self.content_display.setTextCursor(cursor)
            return
            
        # Highlight all occurrences
        self.content_display.moveCursor(QTextCursor.MoveOperation.Start)
        color = QColor(255, 255, 0, 127)  # Semi-transparent yellow
        
        while self.content_display.find(text):
            cursor = self.content_display.textCursor()
            format = cursor.charFormat()
            format.setBackground(color)
            cursor.mergeCharFormat(format)
            
    def show_go_to_dialog(self):
        """Show dialog to jump to specific page"""
        if not self.json_data:
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Go to Page")
        layout = QVBoxLayout()
        
        spin = QSpinBox()
        spin.setMinimum(1)
        spin.setMaximum(self.total_pages)
        spin.setValue(self.current_page)
        layout.addWidget(QLabel("Enter page number:"))
        layout.addWidget(spin)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.go_to_page(spin.value())
            
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
            
            if self.json_data:
                self.total_pages = len(self.json_data)
                first_record = self.json_data[0]
                filename = first_record.get('filename', 'Unknown')
                doc_id = first_record.get('document_id', 'Unknown')
                
                # Update UI
                self.info_label.setText(
                    f"File: {os.path.basename(file_path)} | "
                    f"Document: {filename} | "
                    f"ID: {doc_id}"
                )
                
                # Update page controls
                self.page_spin.setMaximum(self.total_pages)
                self.page_label.setText(f"/ {self.total_pages}")
                self.quick_start.setMaximum(self.total_pages)
                self.quick_end.setMaximum(self.total_pages)
                self.quick_chapter_num.setValue(0)  # Start with 0 for front matter
                
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
        
        if self.json_data and 'chapter_number' in self.json_data[0]:
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
                        
            for ch_data in sorted(chapters_dict.values(), key=lambda x: x['number']):
                chapter = ChapterDefinition(
                    chapter_number=ch_data['number'],
                    start_page=ch_data['start'],
                    end_page=ch_data['end'],
                    chapter_name=ch_data['name']
                )
                self.chapters.append(chapter)
                
            self.update_chapter_table()
            self.update_quick_chapter_number()
            self.status_bar.showMessage(f"Loaded {len(self.chapters)} existing chapters")
            
    def go_to_page(self, page_num: int):
        """Navigate to a specific page"""
        if not self.json_data:
            return
            
        page_num = max(1, min(page_num, self.total_pages))
        self.current_page = page_num
        
        # Update spinner without triggering signal
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(page_num)
        self.page_spin.blockSignals(False)
        
        # Get page data
        page_data = self.json_data[page_num - 1]
        
        # Get content and extract header/footer tags
        content = page_data.get('content', '')
        
        # Extract PageHeader and PageFooter tags
        header_text = ""
        footer_text = ""
        cleaned_content = content
        
        # Extract PageHeader
        header_match = re.search(r'<!--\s*PageHeader\s*[=:]\s*["\']?([^"\'>\n]+)["\']?\s*-->', content, re.IGNORECASE)
        if header_match:
            header_text = f"Header: {header_match.group(1)}"
            # Remove the header tag from content
            cleaned_content = re.sub(r'<!--\s*PageHeader[^>]*-->\s*\n?', '', cleaned_content, flags=re.IGNORECASE)
        
        # Extract PageFooter
        footer_match = re.search(r'<!--\s*PageFooter\s*[=:]\s*["\']?([^"\'>\n]+)["\']?\s*-->', content, re.IGNORECASE)
        if footer_match:
            footer_text = f"Footer: {footer_match.group(1)}"
            # Remove the footer tag from content
            cleaned_content = re.sub(r'<!--\s*PageFooter[^>]*-->\s*\n?', '', cleaned_content, flags=re.IGNORECASE)
        
        # Update displays
        self.page_header_label.setText(header_text)
        self.page_footer_label.setText(footer_text)
        self.content_display.setPlainText(cleaned_content)
        
        # Hide header/footer labels if empty
        self.page_header_label.setVisible(bool(header_text))
        self.page_footer_label.setVisible(bool(footer_text))
        
        # Update page reference
        page_ref = page_data.get('page_reference', '-')
        self.page_ref_label.setText(str(page_ref) if page_ref else '-')
        
        # Update page info
        self.page_info_label.setText(f"Page {page_num} of {self.total_pages}")
        
        # Update chapter indicator
        self.update_chapter_indicator()
        
    def update_chapter_indicator(self):
        """Update the chapter indicator for current page"""
        for chapter in self.chapters:
            if chapter.start_page <= self.current_page <= chapter.end_page:
                self.chapter_indicator.setText(
                    f"  |  Chapter {chapter.chapter_number}: {chapter.chapter_name}"
                )
                return
        self.chapter_indicator.setText("  |  [Unassigned]")
        
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
            self.update_quick_chapter_number()
            self.mark_unsaved_changes()
            
    def quick_add_chapter(self):
        """Quick add a chapter from the quick add fields"""
        if not self.quick_name.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a chapter name")
            self.quick_name.setFocus()
            return
            
        new_chapter = ChapterDefinition(
            chapter_number=self.quick_chapter_num.value(),
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
        
        # Clear and prepare for next
        self.quick_name.clear()
        self.quick_start.setValue(self.quick_end.value() + 1)
        self.quick_end.setValue(min(self.quick_start.value() + 20, self.total_pages))
        self.update_quick_chapter_number()
        self.quick_name.setFocus()
        
    def update_quick_chapter_number(self):
        """Auto-increment the quick chapter number"""
        if self.chapters:
            max_num = max(ch.chapter_number for ch in self.chapters)
            self.quick_chapter_num.setValue(max_num + 1)
        else:
            # Start with 0 if no chapters exist (for front matter)
            self.quick_chapter_num.setValue(0)
            
    def edit_selected_chapter(self):
        """Edit the selected chapter"""
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
                self.update_quick_chapter_number()
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
                self.update_quick_chapter_number()
                self.mark_unsaved_changes()
                
    def update_chapter_table(self):
        """Update the chapter table display"""
        self.chapter_table.setRowCount(len(self.chapters))
        
        for row, chapter in enumerate(self.chapters):
            # Chapter number
            item = QTableWidgetItem(str(chapter.chapter_number))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chapter_table.setItem(row, 0, item)
            
            # Chapter name (with text wrapping)
            name_item = QTableWidgetItem(chapter.chapter_name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.chapter_table.setItem(row, 1, name_item)
            
            # Start page (clickable)
            item = QTableWidgetItem(str(chapter.start_page))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setToolTip("Click to jump to this page")
            item.setForeground(QColor(33, 150, 243))  # Blue color for clickable
            self.chapter_table.setItem(row, 2, item)
            
            # End page (clickable)
            item = QTableWidgetItem(str(chapter.end_page))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setToolTip("Click to jump to this page")
            item.setForeground(QColor(33, 150, 243))  # Blue color for clickable
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
                    
    def auto_detect_chapters(self):
        """Auto-detect chapters using original logic with enhanced name extraction"""
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
        
        # Use original detection logic: look for pages with "Chapter" in the first 500 chars
        chapter_pages = []
        for i, page_data in enumerate(self.json_data):
            content = page_data.get('content', '')
            # Look for chapter markers in the beginning of content (original logic)
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
            
            # Extract chapter name from the content of the start page
            page_content = self.json_data[start_page - 1].get('content', '')
            chapter_name = self.extract_chapter_name(page_content)
            if not chapter_name:
                chapter_name = f"Chapter {i + 1}"
                
            chapter = ChapterDefinition(
                chapter_number=i + 1,
                start_page=start_page,
                end_page=end_page,
                chapter_name=chapter_name
            )
            self.chapters.append(chapter)
            
        self.update_chapter_table()
        self.update_quick_chapter_number()
        self.mark_unsaved_changes()
        QMessageBox.information(
            self, "Auto-Detect Complete",
            f"Detected {len(self.chapters)} chapters"
        )
        
    def validate_chapters(self):
        """Validate all chapter definitions"""
        if not self.chapters:
            QMessageBox.information(self, "Validation", "No chapters defined")
            return
            
        errors = []
        warnings = []
        
        # Individual validation
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
                    f"Gap: pages {gap_start}-{gap_end} between chapters "
                    f"{sorted_chapters[i].chapter_number} and {sorted_chapters[i+1].chapter_number}"
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
                msg += "❌ ERRORS:\n" + "\n".join(errors) + "\n\n"
            if warnings:
                msg += "⚠️ WARNINGS:\n" + "\n".join(warnings)
            QMessageBox.warning(self, "Validation Results", msg)
        else:
            QMessageBox.information(self, "Validation Results", "✅ All chapters valid!")
            
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
            # Apply chapter assignments
            for record in self.json_data:
                page_num = record['page_number']
                
                chapter_assigned = False
                for chapter in self.chapters:
                    if chapter.start_page <= page_num <= chapter.end_page:
                        record['chapter_number'] = chapter.chapter_number
                        record['chapter_name'] = chapter.chapter_name
                        chapter_assigned = True
                        break
                        
                if not chapter_assigned:
                    record.pop('chapter_number', None)
                    record.pop('chapter_name', None)
                    
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
                
            self.unsaved_changes = False
            self.setWindowTitle(f"Chapter Assignment Tool - {os.path.basename(file_path)}")
            self.status_bar.showMessage(f"Saved to {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", "Chapter assignments saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
            
    def mark_unsaved_changes(self):
        """Mark that there are unsaved changes"""
        self.unsaved_changes = True
        title = "Chapter Assignment Tool"
        if self.json_file_path:
            title += f" - {os.path.basename(self.json_file_path)}*"
        else:
            title += "*"
        self.setWindowTitle(title)
        
    def update_ui_state(self):
        """Update UI elements based on current state"""
        has_data = bool(self.json_data)
        
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