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

    # for i in range(len(sentence) - n + 1):
    word_ngrams = Counter([tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)])
    print(f"sentence n-gram is {word_ngrams}")

    # Find the n-gram for refs
    ref_ngrams = [Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]) for ref in references]
    print(f"ref n-gram {ref_ngrams}")

    sentence_count = sum(word_ngrams.values())
    print(sentence_count)

     # Calculate lengths of all reference translations
    ref_lengths = [len(ref) for ref in references]
    print(f"all ref translations length is {ref_lengths}")

    # Find the reference length that is closest to the candidate length
    closest_ref_count = min(ref_lengths, key=lambda ref_len: (abs(ref_len - sentence_count), ref_len))
    print(f"ref closest length is {closest_ref_count}")

    # Count clipped n-grams
    clipped_counts = 0
    for ngram, count in word_ngrams.items():
        max_ref_count = max(ref_ngram_counts.get(ngram, 0) for ref_ngram_counts in ref_ngrams)
        clipped_counts += min(count, max_ref_count)
    print(f"clipped count is {clipped_counts}")

    # precision
    precision = clipped_counts / sentence_count if sentence_count > 0 else 0
    print(f"precision is {precision}")

    # berivity penalty
    brevity_penalty = min(1.0, sentence_count / closest_ref_count) if closest_ref_count > 0 else 0
    print(f"brevity is {brevity_penalty}")

    # bleu score
    bleu_score = brevity_penalty * precision
    return bleu_score
