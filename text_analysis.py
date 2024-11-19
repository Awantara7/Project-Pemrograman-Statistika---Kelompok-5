
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

# Contoh data teks
texts = [
    "Teknologi IoT semakin berkembang pesat di era digital.",
    "Privasi data menjadi perhatian utama di tengah adopsi teknologi baru.",
    "Bagaimana pengguna dapat melindungi data pribadi mereka?",
    "Edukasi tentang keamanan IoT sangat diperlukan."
]

# Pembersihan data teks
def clean_text(text):
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    return text

cleaned_texts = [clean_text(text) for text in texts]

# Tokenisasi
nltk.download('punkt')
tokenized_texts = [word_tokenize(text) for text in cleaned_texts]

# Penghapusan stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

processed_texts = [
    [word for word in tokens if word not in stop_words]
    for tokens in tokenized_texts
]

# Gabungkan semua teks menjadi satu
all_words = [word for tokens in processed_texts for word in tokens]

# Hitung frekuensi kata
freq_dist = FreqDist(all_words)
print("\nFrekuensi Kata:")
print(freq_dist.most_common(10))  # Kata paling sering muncul

# Visualisasi WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Tampilkan matriks fitur
df_vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("\nMatriks Fitur (CountVectorizer):")
print(df_vectorized)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)

# Tampilkan matriks fitur
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nMatriks Fitur (TF-IDF):")
print(df_tfidf)

# Analisis sentimen untuk setiap teks
from textblob import TextBlob
for text in cleaned_texts:
    analysis = TextBlob(text)
    print(f"Teks: {text}")
    print(f"Sentimen: {analysis.sentiment}")
