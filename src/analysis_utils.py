# src/analysis_utils.py
from nltk.metrics.distance import edit_distance # For word-level, need to tokenize first
from nltk.util import ngrams as nltk_ngrams # Renamed to avoid conflict
from src.utils import tokenize_text

def word_level_edit_distance(text1, text2):
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    return edit_distance(tokens1, tokens2)

def common_ngrams_count(text1, text2, n):
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    ngrams1 = set(nltk_ngrams(tokens1, n))
    ngrams2 = set(nltk_ngrams(tokens2, n))
    
    return len(ngrams1.intersection(ngrams2))

def longest_common_subsequence(text1, text2):
    # Word level LCS
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    m = len(tokens1)
    n = len(tokens2)
    
    # L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    L = [[0] * (n + 1) for _ in range(m + 1)] 

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif tokens1[i-1] == tokens2[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]