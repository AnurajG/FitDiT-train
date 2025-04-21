import os
import requests
import base64
from io import BytesIO
import time
from PIL import Image
import json

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def classify_garment_with_gpt4o(image):
    """
    Use GPT-4o to classify a garment image into one of three categories:
    Upper-body, Lower-body, or Dresses
    
    Args:
        image (PIL.Image): The garment image to classify
        
    Returns:
        str: One of "Upper-body", "Lower-body", or "Dresses"
    """
    # Resize image to reduce API payload size
    max_size = 800
    image_resized = image.copy()
    if max(image.size) > max_size:
        image_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # Convert image to base64
    base64_image = image_to_base64(image_resized)
    
    # Configure API request
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system", 
                "content": "You are a clothing category classifier. Your task is to classify the garment in the image into exactly one of these three categories: 'Upper-body', 'Lower-body', or 'Dresses'. Upper-body includes shirts, t-shirts, tops, jackets, coats, and blouses. Lower-body includes pants, jeans, shorts, and skirts. Dresses includes dresses, gowns, and full-body outfits. Respond with ONLY the category name, nothing else."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What category does this garment belong to?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 10
    }
    
    # Make API request with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                
                # Normalize the response
                if "upper" in result.lower() or "top" in result.lower() or "shirt" in result.lower() or "blouse" in result.lower() or "jacket" in result.lower():
                    return "Upper-body"
                elif "lower" in result.lower() or "pant" in result.lower() or "jean" in result.lower() or "skirt" in result.lower() or "short" in result.lower():
                    return "Lower-body"
                elif "dress" in result.lower() or "gown" in result.lower() or "full" in result.lower():
                    return "Dresses"
                else:
                    # Default to Upper-body if unclear
                    return "Upper-body"
            
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit hit, waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
                
            else:
                print(f"API error: {response.status_code}")
                print(response.text)
                return "Upper-body"  # Default fallback
                
        except Exception as e:
            print(f"Error classifying image: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return "Upper-body"  # Default fallback
    
    return "Upper-body"  # Default fallback if all attempts fail

def get_category_from_cache(image_path, category_dir):
    """
    Check if category has already been determined for this image
    
    Args:
        image_path (str): Path to the image file
        category_dir (str): Directory containing category cache files
        
    Returns:
        str or None: Category if found, None otherwise
    """
    # Create cache filename based on original image path
    basename = os.path.basename(image_path)
    cache_path = os.path.join(category_dir, f"{os.path.splitext(basename)[0]}.txt")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read().strip()
    return None

def save_category_to_cache(image_path, category, category_dir):
    """
    Save determined category to cache file
    
    Args:
        image_path (str): Path to the image file
        category (str): Determined category
        category_dir (str): Directory to save category cache files
    """
    # Create cache filename based on original image path
    basename = os.path.basename(image_path)
    cache_path = os.path.join(category_dir, f"{os.path.splitext(basename)[0]}.txt")
    
    with open(cache_path, 'w') as f:
        f.write(category)

def get_garment_category(garment_img, image_path, category_dir):
    """
    Determine garment category, checking cache first then using GPT-4o
    
    Args:
        garment_img (PIL.Image): The garment image
        image_path (str): Path to the image file
        category_dir (str): Directory for category cache
        
    Returns:
        str: One of "Upper-body", "Lower-body", or "Dresses"
    """
    # Ensure category directory exists
    os.makedirs(category_dir, exist_ok=True)
    
    # Check if category is already cached
    cached_category = get_category_from_cache(image_path, category_dir)
    if cached_category:
        print(f"  Using cached category: {cached_category}")
        return cached_category
    
    # Use GPT-4o for classification
    try:
        category = classify_garment_with_gpt4o(garment_img)
        print(f"  Classified with GPT-4o as: {category}")
        
        # Save to cache
        save_category_to_cache(image_path, category, category_dir)
        return category
        
    except Exception as e:
        print(f"  Classification failed: {e}")
        # Default to Upper-body as fallback
        default_category = "Upper-body"
        save_category_to_cache(image_path, default_category, category_dir)
        return default_category