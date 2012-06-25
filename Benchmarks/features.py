
def train_bag(text, n=50):
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
