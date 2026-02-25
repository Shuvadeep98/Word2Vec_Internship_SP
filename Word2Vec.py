import numpy as np
import re
from collections import Counter

# ==========================
# 1: Load and preprocess text
# ==========================
#with open('input.txt', 'r') as f:
with open('Input-2.txt', 'r', encoding="utf-8", errors="ignore") as f:
    text = f.read().lower()

# Simple word-level tokenization (keep alphabetic words)
words = re.findall(r'\b[a-z0-9]+\b', text)
word_counts = Counter(words)
vocab = [w for w in word_counts]

# Mappings
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Convert corpus to ID sequence
data = [word_to_id[w] for w in words if w in word_to_id]

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print(f"Total training tokens: {len(data)}")

# ==========================
# 2: Generate skip-gram pairs
# ==========================
window_size = 5
pairs = []
for i in range(window_size, len(data) - window_size):
    center = data[i]
    context_ids = data[i-window_size:i] + data[i+1:i+window_size+1]
    for context in context_ids:
        pairs.append((center, context))

print(f"Total training pairs: {len(pairs)}")

# Shuffle pairs
np.random.shuffle(pairs)

# ==========================
# 3: Utilities
# ==========================
def sigmoid(x):
    # Numerical stability
    x = np.clip(x, -6, 6)
    return 1 / (1 + np.exp(-x))

def get_negative_samples(true_context, vocab_size, k=5):
    negatives = np.random.randint(0, vocab_size, size=k)
    negatives = [n for n in negatives if n != true_context]
    return negatives

# ==========================
# 4: Initialize embeddings
# ==========================
embedding_dim = 100
np.random.seed(42)
W_in  = np.random.randn(vocab_size, embedding_dim) * 0.1
W_out = np.random.randn(vocab_size, embedding_dim) * 0.1

# ==========================
# 5: Training loop
# ==========================
learning_rate = 0.1
batch_size = 1000
num_steps = 100000  # first 50k pairs

batch_loss = 0

for step, (center_id, context_id) in enumerate(pairs[:num_steps]):

    v_c = W_in[center_id]        # Center word vector
    v_o = W_out[context_id]      # Positive context vector

    # Negative samples
    negatives = get_negative_samples(context_id, vocab_size, k=5)

    # ---------- Forward pass ----------
    pos_score = np.dot(v_c, v_o)
    pos_loss = -np.log(sigmoid(pos_score))

    neg_loss = 0
    for neg_id in negatives:
        v_n = W_out[neg_id]
        neg_score = np.dot(v_c, v_n)
        neg_loss += -np.log(sigmoid(-neg_score))

    loss = pos_loss + neg_loss
    batch_loss += loss

    # ---------- Gradients ----------
    # Positive
    grad_pos = sigmoid(pos_score) - 1
    W_in[center_id] -= learning_rate * grad_pos * v_o
    W_out[context_id] -= learning_rate * grad_pos * v_c

    # Negative samples
    for neg_id in negatives:
        v_n = W_out[neg_id]
        neg_score = np.dot(v_c, v_n)
        grad_neg = sigmoid(neg_score)
        W_in[center_id] -= learning_rate * grad_neg * v_n
        W_out[neg_id] -= learning_rate * grad_neg * v_c

    # ---------- Print average loss per batch ----------
    if (step + 1) % batch_size == 0:
        avg_loss = batch_loss / batch_size
        print(f"Step {step+1}, Avg Loss: {avg_loss:.4f}")
        batch_loss = 0

# ==========================
#Finding Similarity
# ==========================
import numpy as np

# Choose which embedding to use for similarity: W_in or average of W_in & W_out
embeddings = W_in  # Using input embeddings

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def nearest_words(word, top_k=5):
    """Return top_k nearest words to the given word."""
    if word not in word_to_id:
        print(f"'{word}' not in vocabulary!")
        return
    
    word_id = word_to_id[word]
    word_vec = embeddings[word_id]
    
    sims = []
    for i in range(len(embeddings)):
        if i == word_id:
            continue
        sim = cosine_similarity(word_vec, embeddings[i])
        sims.append((id_to_word[i], sim))
    
    sims.sort(key=lambda x: x[1], reverse=True)
    print(f"Nearest words to '{word}':")
    for w, s in sims[:top_k]:
        print(f"  {w} (cos={s:.3f})")

# Example usage:
nearest_words("war")
nearest_words("guns")
nearest_words("night")
# ==========================
#Visualization
# ==========================
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

top_words = [w for w, _ in word_counts.most_common(200)]
indices = [word_to_id[w] for w in top_words]
emb_to_plot = W_in[indices]

pca = PCA(n_components=2)
coords = pca.fit_transform(emb_to_plot)

plt.figure(figsize=(14,10))
plt.scatter(coords[:,0], coords[:,1])

for i, w in enumerate(top_words):
    plt.text(coords[i,0], coords[i,1], w, fontsize=9)

plt.title("Top 200 Word Embeddings Visualized (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
