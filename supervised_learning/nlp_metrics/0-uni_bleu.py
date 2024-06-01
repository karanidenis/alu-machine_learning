#!/usr/bin/env python3

"""This module has a function that
calculates the Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """calculate the unigram BLEU score
    references - list of reference translations
    reference translation - list of words in translation
    sentence - list of model proposed sentence
    """
    sentence_tokens = [word.split() for word in sentence]
    # print(sentence_tokens)
    ref_tokens_list = references
    # print(ref_tokens_list)
    
    word_count = len(sentence_tokens)
    # print(f"sentence unigram is {word_count}")

    # Find the reference length that is closest to the candidate length
    ref_lengths = [len(ref) for ref in ref_tokens_list]

    closest_ref_count = ref_lengths[0]
    min_diff = abs(ref_lengths[0] - word_count)
    
    for ref_length in ref_lengths:
        diff = abs(ref_length - word_count)
        if diff < min_diff:
            min_diff = diff
            closest_ref_count = ref_length
        elif diff == min_diff:
            closest_ref_count = min(closest_ref_count, ref_length)
    # print(f"ref unigram is {closest_ref_count}")

    # Count clipped unigrams
    clipped_counts = 0
    sentence_unigram_counts = {}
    for token in sentence_tokens:
        for word in token:
            if word in sentence_unigram_counts.keys():
                sentence_unigram_counts[word] += 1
            else:
                sentence_unigram_counts[word] = 1
    # print(sentence_unigram_counts)

    for token, count in sentence_unigram_counts.items():
        max_ref_count = max([ref_tokens.count(token) for ref_tokens in ref_tokens_list])
        clipped_counts += min(count, max_ref_count)

    # precision
    precision = clipped_counts / word_count

    # berivity penalty
    if word_count > closest_ref_count:
        return 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_count / word_count)

    # bleu score
    bleu_score = brevity_penalty * precision
    return bleu_score
