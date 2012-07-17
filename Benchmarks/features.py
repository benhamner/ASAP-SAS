import itertools
import re

def generate_letters(num_letters):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters = ["".join(x) for x in itertools.product(alphabet, repeat=num_letters)]
    return letters

def train_bag(text, n=500):
    words = [w for w in text.lower().split(" ") if w]
    word_counts = {}
    for w in words:
        if w not in word_counts:
            word_counts[w] = 0.0
        word_counts[w] += 1.0

    sorted_words = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)
    return sorted_words[:n]

def bag_representation(bag, text):
    return [float(w in text) for w in bag]

def bag_count_representation(bag, text):
    return [float(len(re.findall(w, text))) for w in bag]