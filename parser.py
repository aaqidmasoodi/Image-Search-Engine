import json
import os
import re
from typing import Dict, Any
from preprocess import preprocess_text

def parse_image_metadata(json_file: str) -> Dict[str, Dict[str, Any]]:
    """Parses image metadata into searchable documents"""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Metadata file not found: {json_file}")
    
    try:
        with open(json_file) as f:
            metadata = json.load(f)
        
        documents = {}
        for idx, img in enumerate(metadata):
            doc_id = str(idx)
            filename = img["filename"]
            
            # Extract meaningful title
            title = (img.get("alt_text", "") or 
                    os.path.splitext(filename)[0].replace('_', ' ').title())
            
            # Clean and combine text fields
            caption = re.sub(r'\[\d+\]|\{\{.*?\}\}', '', img.get("caption", ""))
            surrounding = re.sub(r'\[\d+\]|\{\{.*?\}\}', '', img.get("surrounding_text", ""))
            text = " ".join(filter(None, [caption, surrounding]))
            
            documents[doc_id] = {
                "title": title[:200],  # Truncate long titles
                "text": text[:1000],    # Limit text length
                "filename": filename,
                "image_url": img["image_url"],
                "filetype": filename.split('.')[-1].lower()
            }
        
        print(f"✅ Parsed {len(documents)} images from {json_file}")
        return documents
        
    except Exception as e:
        print(f"❌ Error parsing {json_file}: {e}")
        return {}