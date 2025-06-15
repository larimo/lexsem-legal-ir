# src/utils.py
import json
import pickle

def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Add any text preprocessing if needed (though paper implies text is used as-is)
def tokenize_text(text): # Basic tokenizer for BM25/TF-IDF
    return text.lower().split()