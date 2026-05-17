"""
Amaç: 
    - Bu uygulama Bert modelini kullanarak metin benzerliği analizi gerçekleştirecek.
    - Bir sorgu cümlesi (query) ile bir dizi belgenenin (documents) benzerliğini hesaplayarak, en benzer belgeleri bulacak.
    - Her bir BERT modelinden elde edilen embedding (vektör temsili) ile temsil edilir
    - Cosine similarity kullanarak sorgu ve belgeler arasındaki benzerliği hesaplar

Adımlar:
    1. Gerekli kütüphaneleri yükleyin
    2. Önceden eğitilmiş bir BERT modeli ve tokenizer'ı yükleyin
    3. Örnek belgeleri ve sorgu cümlesini tanımlayın
    4. Belgeleri ve sorguyu tokenize edin ve BERT modelini kullanarak embedding'lerini çıkarın
    5. Cosine similarity kullanarak sorgu ve belgeler arasındaki benzerliği hesaplayın
    6. En benzer belgeleri sıralayın ve yazdırın

kurulumlar:
    pip install transformers torch scikit-learn numpy
"""

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_name = "bert-base-uncased" # küçük boyutlu ing metinler

tokenizer = BertTokenizer.from_pretrained(model_name) # modelin tokenizer'ını yükler, bu tokenizer metni modele uygun şekilde tokenize etmek için kullanılır

model = BertModel.from_pretrained(model_name) # modelin kendisini yükler, bu model metin benzerliği analizi için kullanılacak

documents = [
    "Machine learning is a field of artifical intelligence",
    "Deep learning is a subset of machine learning",
    "Artificial intelligence encompasses machine learning and natural language processing (NLP)",
    "Natural language processing involves understanding human language",
    "Data science combines statistics, data analysis, and machine learning"
]

# kullancının sorgusu

query = "What is deep learning?"

def get_embedding(text):

    """
    verilen metni tokenize eder, BERT modelini kullanarak embedding'ini çıkarır ve döndürür
    - tokenization
    - model çalıştırılır
    - embedding çıkarılır
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)  # paddding giriş uzunluğunu eşitler

    # modeli çalıştır
    outputs = model(**inputs)

    # son gizli katmandan ortalama alınarak vektör elde edilir
    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state.mean(dim=1)

    return embedding.detach().numpy() # embedding'i numpy array formatında döndürür

# belgelerin embedding'lerini çıkar
document_embeddings = np.vstack([get_embedding(doc) for doc in documents]) # her belge için embedding çıkarır ve bunları birleştirerek tek bir numpy array oluşturur

query_embedding = get_embedding(query) # sorgunun embedding'ini çıkarır

# benzerliği hesapla
similarities = cosine_similarity(query_embedding, document_embeddings) # sonuç 1 ise sorgu ve belge benzer, 0 ise tamamen farklı

#sonuçları yazdır

for i, score in enumerate(similarities[0]):
    print(f"Document: {documents[i]} - Similarity Score: {score:.4f}")

# en yüksek benzerlik skoruna sahip belgeyi bul
most_similar_index = similarities.argmax() # en yüksek benzerlik skoruna sahip belgenin indeksini bulur

print(f"\nMost similar document to the query: '{query}' is:\n'{documents[most_similar_index]}' with a similarity score of {similarities[0][most_similar_index]:.4f}")
