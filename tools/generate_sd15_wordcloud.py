#!/usr/bin/env python3
"""
Generate word cloud from SD1.5 prompts in llava dataset
"""

import json
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def extract_sd15_prompts(json_file_path):
    """Extract prompts from SD1.5 entries in the dataset"""
    prompts = []
    
    print(f"Reading JSON file: {json_file_path}")
    
    # Read the large JSON file in chunks or line by line
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Total entries in dataset: {len(data)}")
        
        sd15_count = 0
        for entry in data:
            # Check if this entry is related to SD1.5
            image_path = entry.get('image', '')
            if 'sd1.5' in image_path.lower() or 'sd_1.5' in image_path.lower():
                # Extract the prompt from conversations
                conversations = entry.get('conversations', [])
                for conv in conversations:
                    if conv.get('from') == 'gpt':
                        prompt = conv.get('value', '')
                        if prompt:
                            prompts.append(prompt)
                            sd15_count += 1
                            break
        
        print(f"Found {sd15_count} SD1.5 prompts")
        
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []
    
    return prompts

def clean_text(text):
    """Clean and preprocess text for word cloud"""
    # Remove common prompt prefixes/suffixes
    text = re.sub(r'^(A |An |The )', '', text, flags=re.IGNORECASE)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s,.-]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def generate_wordcloud(prompts, output_path='sd15_wordcloud.png', max_words=100):
    """Generate word cloud from prompts"""
    if not prompts:
        print("No prompts found!")
        return
    
    # Combine all prompts
    all_text = ' '.join(prompts)
    
    # Clean the text
    cleaned_text = clean_text(all_text)
    
    print(f"Total text length: {len(cleaned_text)} characters")
    
    # Common words to exclude
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'image', 'photo', 'picture', 'scene', 'view', 'shot'
    }
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=max_words,
        stopwords=stopwords,
        colormap='viridis',
        relative_scaling=0.5,
        random_state=42
    ).generate(cleaned_text)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('SD1.5 Prompts Word Cloud', fontsize=20, pad=20)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Word cloud saved to: {output_path}")
    
    # Show most common words
    words = cleaned_text.lower().split()
    word_freq = Counter(word for word in words if word not in stopwords and len(word) > 2)
    
    print("\nTop 20 most common words:")
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")
    
    return wordcloud

def main():
    parser = argparse.ArgumentParser(description='Generate word cloud from SD1.5 prompts')
    parser.add_argument('--input', '-i', 
                       default='/Users/wangyabin/Downloads/llava_dataset.json',
                       help='Path to the llava dataset JSON file')
    parser.add_argument('--output', '-o', 
                       default='sd15_wordcloud.png',
                       help='Output path for the word cloud image')
    parser.add_argument('--max-words', '-m', 
                       type=int, default=100,
                       help='Maximum number of words in the word cloud')
    
    args = parser.parse_args()
    
    # Extract SD1.5 prompts
    print("Extracting SD1.5 prompts...")
    prompts = extract_sd15_prompts(args.input)
    
    if not prompts:
        print("No SD1.5 prompts found!")
        return
    
    # Generate word cloud
    print("Generating word cloud...")
    generate_wordcloud(prompts, args.output, args.max_words)
    
    print("Done!")

if __name__ == '__main__':
    main()