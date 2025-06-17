# Stage 3 Output Visualization

## Input: Stage 2 JSON Structure
```json
{
  "document_name": "2023_annual_report.pdf",
  "total_pages": 3,
  "pages": [
    {
      "page_number": 1,
      "markdown_content": "# Annual Report 2023\n\n## Executive Summary\n\nThis year marked significant growth..."
    },
    {
      "page_number": 2,
      "markdown_content": "## Financial Performance\n\n| Quarter | Revenue | Growth |\n|---------|---------|--------|\n| Q1 | $1.2M | 15% |..."
    },
    {
      "page_number": 3,
      "markdown_content": "## Future Outlook\n\nWe anticipate continued expansion in key markets..."
    }
  ]
}
```

## Output: Three JSON Files

### 1. **3A_catalog_entries.json** (Document-level metadata)
```json
[
  {
    "document_source": "internal_esg",
    "document_type": "POLICY_DOCUMENT",
    "document_name": "2023_annual_report.pdf",
    "document_description": "Annual report covering 2023 financial performance, strategic initiatives, and future outlook for stakeholders.",
    "document_usage": "Comprehensive 2023 annual report containing: executive summary with year highlights and achievements; detailed financial performance data including quarterly revenue breakdown (Q1: $1.2M with 15% growth); strategic initiatives overview; market expansion plans; future outlook and growth projections. Useful for understanding company performance, financial metrics, strategic direction, and investment decisions. Key topics include revenue growth, market expansion, operational efficiency, and stakeholder value creation.",
    "document_usage_embedding": [0.0234, -0.0567, 0.0891, ...], // 2000 dimensions
    "document_description_embedding": [0.0123, -0.0456, 0.0789, ...], // 2000 dimensions
    "date_created": "2023-01-15T10:30:00Z",
    "date_last_modified": "2023-12-20T14:45:00Z",
    "file_name": "2023_annual_report.pdf",
    "file_type": ".pdf",
    "file_size": 2457600,
    "file_path": "internal_esg/2023_annual_report.pdf",
    "file_link": "//10.0.0.100/shared/internal_esg/2023_annual_report.pdf",
    "processed_json_path": "path/to/output/internal_esg/2A_processed_files/2023_annual_report/2023_annual_report.json"
  }
]
```

### 2. **3B_content_entries.json** (Page-level content)
```json
[
  {
    "document_source": "internal_esg",
    "document_type": "POLICY_DOCUMENT",
    "document_name": "2023_annual_report.pdf",
    "page_number": 1,
    "section_id": 1,
    "section_name": "2023_annual_report_page_1",
    "section_summary": "Page 1 of 2023_annual_report.pdf",
    "section_content": "# Annual Report 2023\n\n## Executive Summary\n\nThis year marked significant growth across all business units. Our commitment to sustainable practices and innovation has positioned us as a market leader.\n\n### Key Achievements\n- Revenue growth of 23% year-over-year\n- Expansion into 5 new markets\n- Launch of innovative product line\n- Carbon neutral operations achieved",
    "date_created": "2025-01-17T09:30:45Z"
  },
  {
    "document_source": "internal_esg",
    "document_type": "POLICY_DOCUMENT",
    "document_name": "2023_annual_report.pdf",
    "page_number": 2,
    "section_id": 1,
    "section_name": "2023_annual_report_page_2",
    "section_summary": "Page 2 of 2023_annual_report.pdf",
    "section_content": "## Financial Performance\n\n| Quarter | Revenue | Growth | Operating Margin |\n|---------|---------|--------|------------------|\n| Q1 2023 | $1.2M   | 15%    | 22%              |\n| Q2 2023 | $1.4M   | 18%    | 24%              |\n| Q3 2023 | $1.6M   | 22%    | 26%              |\n| Q4 2023 | $1.9M   | 31%    | 28%              |\n\n### Revenue Breakdown by Region\n- North America: 45%\n- Europe: 30%\n- Asia Pacific: 20%\n- Other: 5%",
    "date_created": "2025-01-17T09:30:45Z"
  },
  {
    "document_source": "internal_esg",
    "document_type": "POLICY_DOCUMENT",
    "document_name": "2023_annual_report.pdf",
    "page_number": 3,
    "section_id": 1,
    "section_name": "2023_annual_report_page_3",
    "section_summary": "Page 3 of 2023_annual_report.pdf",
    "section_content": "## Future Outlook\n\nWe anticipate continued expansion in key markets with focus on:\n\n1. **Digital Transformation**: Investing $5M in AI and automation\n2. **Sustainability**: Achieving net-zero emissions by 2025\n3. **Market Expansion**: Entering South American markets\n4. **Innovation**: R&D budget increased by 40%\n\n### 2024 Projections\n- Expected revenue growth: 25-30%\n- New product launches: 8\n- Headcount increase: 150 employees\n- Market share target: 15% (from current 12%)",
    "date_created": "2025-01-17T09:30:45Z"
  }
]
```

### 3. **3C_anonymization_report.json** (DLP/Anonymization tracking)
```json
[
  {
    "document_source": "internal_esg",
    "document_type": "POLICY_DOCUMENT",
    "document_name": "2023_annual_report.pdf",
    "document_description": "Annual report covering 2023 financial performance, strategic initiatives, and future outlook for stakeholders.",
    "document_usage": "Comprehensive 2023 annual report containing...",
    "document_usage_embedding": [0.0234, -0.0567, 0.0891, ...],
    "document_description_embedding": [0.0123, -0.0456, 0.0789, ...],
    "date_created": "2023-01-15T10:30:00Z",
    "date_last_modified": "2023-12-20T14:45:00Z",
    "file_name": "2023_annual_report.pdf",
    "file_type": ".pdf",
    "file_size": 2457600,
    "file_path": "internal_esg/2023_annual_report.pdf",
    "file_link": "//10.0.0.100/shared/internal_esg/2023_annual_report.pdf",
    "processed_json_path": "path/to/output/internal_esg/2A_processed_files/2023_annual_report/2023_annual_report.json",
    "dlp_anonymized_or_flagged": false,
    "dlp_findings": []
  }
]
```

## Key Changes from Previous Version

### Before (with section splitting):
- Page 1 → Section 1, Section 2
- Page 2 → Section 1, Section 2  
- Page 3 → Section 1, Section 2
- **Total: 6 content entries**

### After (page-level only):
- Page 1 → 1 content entry
- Page 2 → 1 content entry
- Page 3 → 1 content entry
- **Total: 3 content entries**

## Benefits:
1. **Simpler structure** - One entry per page
2. **Natural boundaries** - Pages are logical units
3. **Less data** - Fewer rows in database
4. **Faster processing** - No section detection needed
5. **Easier retrieval** - Direct page-to-content mapping