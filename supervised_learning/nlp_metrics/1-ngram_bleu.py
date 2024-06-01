#!/usr/bin/env python3

"""This module has a function that
calculates the n-gram BLEU score"""
import numpy as np
from collections import Counter

def ngram_bleu(references, sentence, n):
    """calculate the n-gram BLEU score
    references - list of reference translations
    reference translation - list of words in translation
    sentence - list of model proposed sentence
    n - size of n-gram for evaluation
    """

    # Generate n-grams for the sentence
    sentence_ngrams = Counter(tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1))
    
    # Generate n-grams for the references
    reference_ngrams = [Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)) for ref in references]
    
    # Count the total number of n-grams in the sentence
    sentence_count = sum(sentence_ngrams.values())
    
    # Find the reference length that is closest to the sentence length
    ref_lengths = [len(ref) for ref in references]
    closest_ref_count = min(ref_lengths, key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len))
    
    # Count the clipped n-grams
    clipped_count = 0
    for ngram, count in sentence_ngrams.items():
        max_ref_count = max(ref_ngram.get(ngram, 0) for ref_ngram in reference_ngrams)
        clipped_count += min(count, max_ref_count)
    
    # Calculate precision
    precision = clipped_count / sentence_count if sentence_count > 0 else 0
    
    # Calculate brevity penalty
    brevity_penalty = 1.0 if len(sentence) > closest_ref_count else np.exp(1 - closest_ref_count / len(sentence))
    
    # Calculate BLEU score
    bleu_score = brevity_penalty * precision
    
    return bleu_score
