"""
PDF Element Extraction Module
Extracts text, tables, and images from PDF documents using PyMuPDF.
"""


import fitz  # PyMuPDF
from typing import Dict, List
from pathlib import Path
from PIL import Image
import io

def extract_images_with_pymupdf(pdf_path: str, output_dir: str = "data/extracted_images") -> List[Dict]:
    """
    Extract images from PDF using PyMuPDF.
    """
    images_dir = Path(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_elements = []
    pdf_document = fitz.open(pdf_path)
    
    print(f"  🖼️  Extracting images with PyMuPDF...")
    
    image_count = 0
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Skip very small images
                if image.size[0] < 50 or image.size[1] < 50:
                    continue
                
                image_filename = f"image_p{page_num+1}_{img_index}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                
                image_elements.append({
                    "path": str(image_path),
                    "page": page_num + 1,
                    "element_id": f"image_{image_count}",
                    "size": image.size
                })
                image_count += 1
            except Exception as e:
                print(f"    ⚠️  Failed to extract image on page {page_num+1}: {e}")
    
    pdf_document.close()
    print(f"    ✅ Extracted {image_count} images")
    return image_elements


def extract_tables_with_pymupdf(pdf_path: str, tables_dir: str) -> List[Dict]:
    """
    Extract tables using PyMuPDF's table detection.
    """
    print(f"  📊 Detecting tables with PyMuPDF...")
    
    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    
    pdf_document = fitz.open(pdf_path)
    table_elements = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Find tables using PyMuPDF
        tabs = page.find_tables()
        
        for tab_index, tab in enumerate(tabs):
            try:
                # Extract table as pandas dataframe then convert to text
                df = tab.to_pandas()
                table_text = df.to_string(index=False)
                
                if table_text and len(table_text) > 20:
                    table_id = len(table_elements)
                    table_elements.append({
                        "text": table_text,
                        "html": df.to_html(index=False),
                        "page": page_num + 1,
                        "element_id": f"table_{table_id}"
                    })
                    
                    # Save table
                    table_path = tables_path / f"table_{table_id}.txt"
                    with open(table_path, 'w', encoding='utf-8') as f:
                        f.write(table_text)
            except Exception as e:
                print(f"    ⚠️  Error processing table on page {page_num+1}: {e}")
    
    pdf_document.close()
    print(f"    ✅ Detected {len(table_elements)} tables")
    return table_elements


def extract_text_pymupdf(pdf_path: str) -> List[Dict]:
    """
    Extract text using PyMuPDF.
    """
    print(f"  📄 Extracting text with PyMuPDF...")
    
    pdf_document = fitz.open(pdf_path)
    text_elements = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Extract text by blocks
        blocks = page.get_text("blocks")
        
        for block in blocks:
            if len(block) >= 5:
                text = block[4].strip()
                
                # Only keep meaningful text blocks
                if text and len(text) > 20:
                    text_elements.append({
                        "text": text,
                        "type": "Text",
                        "page": page_num + 1,
                        "element_id": f"text_p{page_num}_{len([t for t in text_elements if t['page'] == page_num + 1])}"
                    })
    
    pdf_document.close()
    print(f"    ✅ Extracted {len(text_elements)} text blocks")
    return text_elements


def extract_elements(pdf_path: str, output_dir: str = "data") -> Dict[str, List]:
    """
    Extract elements from PDF and separate into text, tables, and images.
    """
    print(f"🔍 Extracting elements from {pdf_path}...")
    
    # Create output directories
    images_dir = Path(output_dir) / "extracted_images"
    tables_dir = Path(output_dir) / "extracted_tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all element types
    image_elements = extract_images_with_pymupdf(pdf_path, str(images_dir))
    table_elements = extract_tables_with_pymupdf(pdf_path, str(tables_dir))
    text_elements = extract_text_pymupdf(pdf_path)
    
    results = {
        "text": text_elements,
        "tables": table_elements,
        "images": image_elements
    }
    
    print(f"\n📊 Extraction Summary:")
    print(f"  - Text elements: {len(text_elements)}")
    print(f"  - Tables: {len(table_elements)}")
    print(f"  - Images: {len(image_elements)}\n")
    
    return results


if __name__ == "__main__":
    # Test extraction
    pdf_path = "docs/attention-is-all-you-need.pdf"
    results = extract_elements(pdf_path)
    print("✨ Extraction complete!")
    if results['text']:
        print(f"\nSample text: {results['text'][0]['text'][:100]}...")
    if results['tables']:
        print(f"\nFound {len(results['tables'])} tables!")
        print(f"Sample table: {results['tables'][0]['text'][:200]}...")
    if results['images']:
        print(f"\nSample image: {results['images'][0]}")
