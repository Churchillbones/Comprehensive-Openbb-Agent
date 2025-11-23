"""
PDF Processor Tool

Extract text and data from PDF documents.
Migrated from: comprehensive_agent/processors/pdf.py
"""

from typing import Dict, Any, Optional
import logging
from comprehensive_agent.processors.pdf import process_pdf_data

logger = logging.getLogger(__name__)


async def process_pdf(
    pdf_data: Any,
    extract_images: bool = False,
    max_pages: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process PDF and extract text content

    Args:
        pdf_data: PDF data (base64 string, file path, or bytes)
        extract_images: Whether to extract images (default: False)
        max_pages: Maximum number of pages to process (default: all)

    Returns:
        Dict with extracted content:
            - status: "success" or "error"
            - text: Extracted text content
            - page_count: Number of pages processed
            - metadata: PDF metadata
    """
    try:
        # Use existing PDF processor
        result = await process_pdf_data(pdf_data)

        # Check if result is an error
        if isinstance(result, str) and "error" in result.lower():
            return {
                "status": "error",
                "error": result,
                "text": "",
                "page_count": 0
            }

        # Parse the result
        text_content = result if isinstance(result, str) else str(result)

        # Count pages (rough estimate based on form feeds or page breaks)
        page_count = text_content.count("\f") + 1 if "\f" in text_content else 1

        logger.info(f"Processed PDF: {page_count} pages, {len(text_content)} characters")

        return {
            "status": "success",
            "text": text_content,
            "page_count": page_count,
            "character_count": len(text_content),
            "metadata": {
                "has_content": bool(text_content.strip()),
                "extract_images": extract_images
            }
        }

    except Exception as e:
        logger.error(f"PDF processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "text": "",
            "page_count": 0
        }
