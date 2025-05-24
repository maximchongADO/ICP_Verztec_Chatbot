import os
import re
import json
import tiktoken

def count_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def chunk_bullet_points(text, min_tokens=200, max_tokens=600, overlap_tokens=100):
    bullets = re.split(r'\n\s*(?:[-*•]|\d+\.)\s+', text)
    bullets = [b.strip() for b in bullets if b.strip()]
    chunks = []
    current_chunk = ""
    current_len = 0

    for bullet in bullets:
        bullet_len = count_tokens(bullet)
        if current_len + bullet_len <= max_tokens:
            if current_chunk:
                current_chunk += "\n" + bullet
            else:
                current_chunk = bullet
            current_len += bullet_len
        else:
            if current_len >= min_tokens:
                chunks.append(current_chunk)
                if overlap_tokens > 0:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens = encoding.encode(current_chunk)
                    overlap_tokens_actual = min(overlap_tokens, len(tokens))
                    overlap_text = encoding.decode(tokens[-overlap_tokens_actual:])
                    current_chunk = overlap_text + "\n" + bullet
                    current_len = count_tokens(current_chunk)
                else:
                    current_chunk = bullet
                    current_len = bullet_len
            else:
                if current_chunk:
                    current_chunk += "\n" + bullet
                else:
                    current_chunk = bullet
                current_len += bullet_len
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def chunk_paragraphs(text, min_tokens=200, max_tokens=600, overlap_tokens=100):
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip().replace('\n', ' ') for p in paragraphs if p.strip()]
    chunks = []
    current_chunk = ""
    current_len = 0

    for para in paragraphs:
        para_len = count_tokens(para)
        if current_len + para_len <= max_tokens:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_len += para_len
        else:
            if current_len >= min_tokens:
                chunks.append(current_chunk)
                if overlap_tokens > 0:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens = encoding.encode(current_chunk)
                    overlap_tokens_actual = min(overlap_tokens, len(tokens))
                    overlap_text = encoding.decode(tokens[-overlap_tokens_actual:])
                    current_chunk = overlap_text + "\n\n" + para
                    current_len = count_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_len = para_len
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_len += para_len
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def is_bullet_point_style(text, check_lines=10):
    lines = text.split('\n')[:check_lines]
    bullet_patterns = [r'^\s*[-*•]\s+', r'^\s*\d+\.\s+']
    bullet_count = 0
    for line in lines:
        if any(re.match(pattern, line) for pattern in bullet_patterns):
            bullet_count += 1
    return bullet_count >= (check_lines // 2)

def batch_chunk_files(input_folder, output_folder, min_tokens=200, max_tokens=600, overlap_tokens=100):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.txt'):
            continue
        filepath = os.path.join(input_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if is_bullet_point_style(text):
            chunks = chunk_bullet_points(text, min_tokens, max_tokens, overlap_tokens)
            style = 'bullet'
        else:
            chunks = chunk_paragraphs(text, min_tokens, max_tokens, overlap_tokens)
            style = 'paragraph'
        out_chunks = []
        for i, chunk in enumerate(chunks):
            out_chunks.append({
                "chunk_index": i,
                "chunk_text": chunk,
                "source_file": filename,
                "chunk_style": style,
                "token_count": count_tokens(chunk)
            })
        out_path = os.path.join(output_folder, filename.replace('.txt', '_chunks.json'))
        with open(out_path, 'w', encoding='utf-8') as outf:
            json.dump({
                "source_file": filename,
                "chunk_style": style,
                "chunks": out_chunks
            }, outf, indent=2)
        print(f"Processed {filename}: {len(chunks)} chunks saved to {out_path}")

# Example usage:
batch_chunk_files(r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-2\data\cleaned", "chunked_output")

