# PDF Chapter Splitting Plan for EY Prep

## Overview
Modify the Chapter Assignment Tool to split the original large PDF (~200MB) into individual chapter PDFs when saving chapter assignments.

## Current Workflow
1. User loads JSON with page-level records
2. User assigns chapters to pages
3. User saves JSON with chapter assignments

## New Workflow
1. User loads JSON with page-level records
2. **NEW: User also selects the original PDF file**
3. User assigns chapters to pages
4. User saves:
   - JSON with chapter assignments (existing)
   - **NEW: Individual chapter PDFs**

## Implementation Plan

### Phase 1: UI Modifications

#### Add PDF Selection
```python
# In ChapterAssignmentTool.__init__
self.pdf_file_path = None
self.pdf_reader = None

# Add menu item or button
def open_pdf_file(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Source PDF", "", "PDF Files (*.pdf)"
    )
    if file_path:
        self.load_pdf_file(file_path)

def load_pdf_file(self, file_path: str):
    try:
        self.pdf_file_path = file_path
        self.pdf_reader = PdfReader(file_path)
        self.status_bar.showMessage(f"Loaded PDF: {os.path.basename(file_path)}")
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")
```

#### Update Info Display
Show both JSON and PDF status:
```python
self.info_label.setText(
    f"JSON: {json_name} | PDF: {pdf_name} | Pages: {total_pages}"
)
```

### Phase 2: PDF Splitting Logic

#### Core Splitting Function
```python
from pypdf import PdfReader, PdfWriter
import os

def split_pdf_by_chapters(self, pdf_path: str, output_dir: str):
    """Split PDF into individual chapter files based on chapter definitions."""
    
    if not self.chapters:
        logging.warning("No chapters defined for splitting")
        return []
    
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return []
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        created_files = []
        
        for chapter in self.chapters:
            # Create writer for this chapter
            writer = PdfWriter()
            
            # Add pages for this chapter (0-indexed in pypdf)
            for page_num in range(chapter.start_page - 1, chapter.end_page):
                if page_num < total_pages:
                    writer.add_page(reader.pages[page_num])
            
            # Generate filename
            safe_name = cleanup_filename(chapter.chapter_name)
            filename = f"{chapter.chapter_number:02d}_{safe_name}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            # Write the chapter PDF
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            created_files.append({
                'chapter_number': chapter.chapter_number,
                'chapter_name': chapter.chapter_name,
                'filename': filename,
                'path': output_path,
                'pages': chapter.end_page - chapter.start_page + 1
            })
            
            logging.info(f"Created chapter PDF: {filename}")
        
        return created_files
        
    except Exception as e:
        logging.error(f"Error splitting PDF: {e}")
        return []

def cleanup_filename(name: str) -> str:
    """Clean chapter name for use as filename."""
    # Remove invalid characters
    name = re.sub(r'[\\/:?*<>|"\']', '', name)
    # Replace spaces/underscores with single underscore
    name = re.sub(r'\s+', '_', name)
    # Limit length
    return name[:100].strip('_')
```

### Phase 3: JSON Updates

#### Update Records with New Filenames
```python
def update_json_with_chapter_files(self, created_files: List[Dict]):
    """Update JSON records to reference the new chapter PDF files."""
    
    # Create mapping of chapter_number to new filename
    chapter_to_file = {
        f['chapter_number']: {
            'filename': f['filename'],
            'filepath': f['path']
        }
        for f in created_files
    }
    
    # Update each record with new filename
    for record in self.json_data:
        chapter_num = record.get('chapter_number')
        if chapter_num in chapter_to_file:
            # Update filename to the new chapter PDF
            record['filename'] = chapter_to_file[chapter_num]['filename']
            record['filepath'] = chapter_to_file[chapter_num]['filepath']
            # Keep original filename for reference
            record['original_filename'] = record.get('filename', 'unknown.pdf')
    
    logging.info("Updated JSON records with new chapter PDF filenames")
```

### Phase 4: NAS Integration

#### Modify Save Workflow
```python
def save_to_json(self):
    """Save chapter assignments and split PDFs."""
    
    # 1. Check if PDF is loaded
    if not self.pdf_file_path:
        reply = QMessageBox.question(
            self, "No PDF Loaded",
            "No source PDF loaded. Save JSON only without creating chapter PDFs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            return
        else:
            # Prompt to load PDF
            self.open_pdf_file()
            if not self.pdf_file_path:
                return
    
    # 3. Create output directory on NAS
    share_name = NAS_PARAMS["share"]
    pdf_dir = os.path.dirname(self.json_file_path)
    
    # Create subfolder for chapter PDFs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"chapters_{timestamp}"
    output_path = os.path.join(pdf_dir, output_folder)
    
    # 4. Split PDF locally first
    with tempfile.TemporaryDirectory() as temp_dir:
        created_files = self.split_pdf_by_chapters(
            self.pdf_file_path, temp_dir
        )
        
        if created_files:
            # 5. Upload to NAS
            progress = QProgressDialog(
                "Uploading chapter PDFs to NAS...", 
                "Cancel", 0, len(created_files), self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            for i, file_info in enumerate(created_files):
                progress.setValue(i)
                if progress.wasCanceled():
                    break
                
                local_path = file_info['path']
                nas_path = os.path.join(output_path, file_info['filename'])
                
                with open(local_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                if not write_to_nas(share_name, nas_path, pdf_bytes):
                    logging.error(f"Failed to upload {file_info['filename']}")
            
            progress.setValue(len(created_files))
            
            # 6. Update JSON with new filenames
            self.update_json_with_chapter_files(created_files)
            
            # 7. Save updated JSON
            self.save_json_to_file(self.json_file_path)
            
            QMessageBox.information(
                self, "Success",
                f"Created {len(created_files)} chapter PDFs in:\n{output_path}\n"
                f"JSON updated with new filenames"
            )
```

#### Example JSON Update

**Before (original):**
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "ey_guide_2024.pdf",  // Original 200MB file
  "filepath": "semantic_search/source_documents/ey/ey_guide_2024.pdf",
  "page_number": 42,
  "chapter_number": 3,
  "chapter_name": "Revenue Recognition",
  "content": "..."
}
```

**After (updated):**
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "03_Revenue_Recognition.pdf",  // New chapter PDF
  "filepath": "semantic_search/source_documents/ey/chapters_20240313_143022/03_Revenue_Recognition.pdf",
  "original_filename": "ey_guide_2024.pdf",  // Preserved for reference
  "page_number": 42,  // Still the original page number from the full PDF
  "chapter_number": 3,
  "chapter_name": "Revenue Recognition",
  "content": "..."
}
```

### Phase 5: File Organization

#### NAS Directory Structure
```
semantic_search/
├── source_documents/
│   └── ey/
│       ├── ey_guide_2024.pdf (original 200MB file)
│       └── chapters_20240313_143022/
│           ├── 00_Front_Matter.pdf
│           ├── 01_Introduction.pdf
│           ├── 02_Revenue_Recognition.pdf
│           ├── 03_Leases.pdf
│           └── ...
└── prep_output/
    └── ey/
        └── ey_prep_output_with_chapters.json
```

#### Naming Convention
- Format: `{chapter_number:02d}_{chapter_name}.pdf`
- Examples:
  - `00_Front_Matter.pdf` (Chapter 0)
  - `01_Introduction.pdf`
  - `15_Financial_Instruments.pdf`

### Phase 5: Downstream Impact

#### Benefits for Processing
1. **Stage 1**: Can process individual chapter PDFs instead of one large file
2. **Memory efficiency**: Smaller files easier to handle
3. **Parallel processing**: Could process chapters concurrently
4. **Error recovery**: If one chapter fails, others unaffected

#### Updated Stage 1 Input
Instead of one large JSON, could have:
- One JSON per chapter
- Each references its specific PDF
- Maintains exact page mapping

### Phase 6: Optional Enhancements

#### 1. Metadata Preservation
```python
# Add metadata to each chapter PDF
writer.add_metadata({
    '/Title': f'Chapter {chapter.chapter_number}: {chapter.chapter_name}',
    '/Subject': 'EY Technical Accounting Guide',
    '/Creator': 'IRIS Chapter Splitter',
    '/ChapterNumber': str(chapter.chapter_number),
    '/OriginalDocument': os.path.basename(pdf_path)
})
```

#### 2. Validation
```python
def validate_chapter_pdfs(self, created_files: List[Dict]):
    """Verify chapter PDFs were created correctly."""
    for file_info in created_files:
        try:
            reader = PdfReader(file_info['path'])
            actual_pages = len(reader.pages)
            expected_pages = file_info['pages']
            
            if actual_pages != expected_pages:
                logging.warning(
                    f"Page count mismatch in {file_info['filename']}: "
                    f"expected {expected_pages}, got {actual_pages}"
                )
        except Exception as e:
            logging.error(f"Failed to validate {file_info['filename']}: {e}")
```

#### 3. Progress Tracking
Show progress during splitting for better UX:
```python
progress_dialog = QProgressDialog(
    "Splitting PDF by chapters...", "Cancel", 
    0, len(self.chapters), self
)
```

## Implementation Priority

1. **High Priority**
   - PDF loading in UI
   - Basic splitting function
   - Save workflow integration

2. **Medium Priority**
   - NAS upload with progress
   - Filename sanitization
   - Error handling

3. **Low Priority**
   - Metadata preservation
   - Validation checks
   - Batch processing options

## Testing Plan

1. Test with small PDF first (few pages)
2. Verify chapter boundaries are correct
3. Test with full 200MB PDF
4. Verify NAS upload works
5. Test error cases (missing pages, invalid chapters)

## Success Criteria

- [x] User can load both JSON and PDF
- [x] Chapters are split correctly by page ranges
- [x] Files are named consistently
- [x] PDFs are uploaded to NAS
- [x] Original PDF is preserved
- [x] Process handles errors gracefully