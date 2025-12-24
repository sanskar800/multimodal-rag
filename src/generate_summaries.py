"""
Summary Generation Module
Generates summaries for text, tables, and images using Llama-8B and Gemini.
"""

import os
from typing import List, Dict
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize API clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def generate_text_summary(text: str, element_id: str = "") -> str:
    """
    Generate summary for text elements using Llama via Groq.
    
    Args:
        text: Text content to summarize
        element_id: Unique identifier for the element
        
    Returns:
        Generated summary
    """
    try:
        prompt = f"""Summarize the following text concisely while preserving key information and context:

Text: {text}

Provide a clear, informative summary in 2-3 sentences."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"⚠️ Error generating text summary for {element_id}: {e}")
        return text[:200]  # Fallback to truncated text


def generate_table_summary(table_text: str, element_id: str = "") -> str:
    """
    Generate summary for table elements using Llama via Groq.
    
    Args:
        table_text: Table content as text
        element_id: Unique identifier for the element
        
    Returns:
        Generated summary
    """
    try:
        prompt = f"""Analyze and summarize the following table, highlighting key data points and relationships:

Table:
{table_text}

Provide a summary that captures the table's structure and main findings."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes tables and extracts key insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"⚠️ Error generating table summary for {element_id}: {e}")
        return table_text[:200]  # Fallback to truncated text


def generate_image_summary(image_path: str, element_id: str = "") -> str:
    """
    Generate summary for image elements using Gemini Flash (cost-effective).
    
    Args:
        image_path: Path to the image file
        element_id: Unique identifier for the element
        
    Returns:
        Generated summary
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Use Gemini 2.5 Flash - latest stable version
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Describe this image in detail. Focus on important visual elements, diagrams, charts, or any text present. Be specific and informative about what you see."
        
        response = model.generate_content([prompt, img])
        
        summary = response.text.strip()
        return summary
    
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Error generating image summary for {element_id}: {error_msg}")
        
        # Provide descriptive fallback
        try:
            img = Image.open(image_path)
            fallback = f"Image {element_id}: {img.size[0]}x{img.size[1]} pixels. (Note: AI description failed: {error_msg[:50]})"
            return fallback
        except:
            return f"Image at {image_path} (Description unavailable)"


def generate_summaries(elements: Dict[str, List]) -> Dict[str, List[Dict]]:
    """
    Generate summaries for all extracted elements.
    
    Args:
        elements: Dictionary with 'text', 'tables', and 'images' lists
        
    Returns:
        Dictionary with summarized elements including metadata
    """
    print("\n📝 Generating summaries...")
    
    summarized_elements = {
        "text": [],
        "tables": [],
        "images": []
    }
    
    # Generate text summaries
    print(f"\n  Processing {len(elements['text'])} text elements...")
    for elem in elements['text']:
        summary = generate_text_summary(elem['text'], elem['element_id'])
        summarized_elements['text'].append({
            "summary": summary,
            "original_text": elem['text'],
            "element_id": elem['element_id'],
            "page": elem['page'],
            "element_type": "text"
        })
    
    # Generate table summaries
    print(f"  Processing {len(elements['tables'])} table elements...")
    for elem in elements['tables']:
        summary = generate_table_summary(elem['text'], elem['element_id'])
        summarized_elements['tables'].append({
            "summary": summary,
            "original_text": elem['text'],
            "element_id": elem['element_id'],
            "page": elem['page'],
            "element_type": "table"
        })
    
    # Generate image summaries
    print(f"  Processing {len(elements['images'])} image elements with Gemini...")
    for elem in elements['images']:
        summary = generate_image_summary(elem['path'], elem['element_id'])
        summarized_elements['images'].append({
            "summary": summary,
            "image_path": elem['path'],
            "element_id": elem['element_id'],
            "page": elem['page'],
            "element_type": "image"
        })
    
    print("  ✅ All summaries generated!\n")
    return summarized_elements


if __name__ == "__main__":
    # Test summary generation
    test_text = "The Transformer architecture uses self-attention mechanisms to process sequences."
    summary = generate_text_summary(test_text, "test_1")
    print(f"Test summary: {summary}")
