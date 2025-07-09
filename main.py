import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# ----------------------------
# 0. Download NLTK data
# ----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ----------------------------
# 1. Load raw documents
# ----------------------------
folder_path = "newsgroups"
documents = []
true_labels = []

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if not filename.startswith('.') and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            split_docs = re.split(r'\nFrom: ', content)
            for doc in split_docs:
                doc = doc.strip()
                if len(doc) > 50:
                    documents.append(doc)
                    true_labels.append(filename)

print(f"✅ Loaded {len(documents)} documents from {len(set(true_labels))} categories.")

# ----------------------------
# 2. Optional: Limit to 5000 docs
# ----------------------------
documents = documents[:5000]
true_labels = true_labels[:5000]
print(f"📉 Reduced to {len(documents)} documents for faster clustering.")

# ----------------------------
# 3. Preprocess: Tokenize & Lemmatize
# ----------------------------
lemmatizer = WordNetLemmatizer()
cleaned_docs = []

for doc in documents:
    tokens = word_tokenize(doc.lower())
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
    cleaned_text = " ".join(lemmatized)
    if cleaned_text:
        cleaned_docs.append(cleaned_text)

print(f"✅ Cleaned docs: {len(cleaned_docs)}")

# ----------------------------
# 4. TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
X = vectorizer.fit_transform(cleaned_docs)
terms = vectorizer.get_feature_names_out()
print(f"✅ TF-IDF shape: {X.shape}")

# ----------------------------
# 5. KMeans with Best K (Silhouette Method)
# ----------------------------
sil_scores = []
K_range = range(2, 21)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
    preds = km.fit_predict(X)
    score = silhouette_score(X, preds)
    sil_scores.append(score)
    print(f"K={k}, Silhouette Score={score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 4))
plt.plot(K_range, sil_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Choose best K (highest silhouette score)
best_k = K_range[np.argmax(sil_scores)]
print(f"✅ Best K chosen: {best_k}")

# Fit KMeans with best K
kmeans = KMeans(n_clusters=best_k, init='k-means++', max_iter=100, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# ----------------------------
# 6. WordClouds for KMeans
# ----------------------------
centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

def get_kmeans_freq(cluster_index):
    term_freq = kmeans.cluster_centers_[cluster_index]
    sorted_terms = centroids[cluster_index]
    return {terms[i]: term_freq[i] for i in sorted_terms[:50]}

def show_wordcloud(freq_dict, title):
    wc = WordCloud(background_color="white", max_words=50)
    wc.generate_from_frequencies(freq_dict)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

for i in range(best_k):
    print(f"\n🔹 KMeans Cluster {i} Keywords:")
    freq_dict = get_kmeans_freq(i)
    print(", ".join(list(freq_dict.keys())[:10]))
    show_wordcloud(freq_dict, f"KMeans Cluster {i}")

# ----------------------------
# 7. LDA Topic Modeling
# ----------------------------
lda = LatentDirichletAllocation(n_components=best_k, max_iter=10, learning_method='online', random_state=42)
lda.fit(X)

def get_lda_freq(topic_idx):
    topic = lda.components_[topic_idx]
    top_terms = topic.argsort()[::-1][:50]
    return {terms[i]: topic[i] for i in top_terms}

for i in range(best_k):
    print(f"\n🔸 LDA Topic {i} Keywords:")
    freq_dict = get_lda_freq(i)
    print(", ".join(list(freq_dict.keys())[:10]))
    show_wordcloud(freq_dict, f"LDA Topic {i}")

# ----------------------------
# 8. t-SNE Visualization of Clusters
# ----------------------------
print("⏳ Reducing dimensionality with t-SNE...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_2d = tsne.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='tab20', s=10)
plt.title("2D Visualization of KMeans Clusters (t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Export to CSV
# ----------------------------
df = pd.DataFrame({'Document': cleaned_docs, 'KMeans Cluster': kmeans_labels})
df.to_csv("kmeans_clustered_documents.csv", index=False)
print("📁 Exported clustered documents to 'kmeans_clustered_documents.csv'")
