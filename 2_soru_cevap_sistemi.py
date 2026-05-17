"""
Amaç:
    - BERT tabanlı bir soru-cevap sistemi oluşturmak
    - Model veirlen bir metin içerisinden belirli bir sorunun cevabını bulur.
    - Huggingface bert-large-uncased-whole-word-masking-finetuned-squad modelini kullanarak, verilen bir metin ve soru için en uygun cevabı üretir.
    Bu model SQuAD (Stanford Question Answering Dataset) üzerinde ince ayar yapılmış bir BERT modelidir ve metin içinden sorulara cevap bulma görevinde oldukça başarılıdır.

Adımlar:
    1. Gerekli kütüphaneleri yükleyin
    2. Önceden eğitilmiş bir soru-cevap modeli ve tokenizer'ı yükleyin
    3. predict_answer fonksiyonunu tanımlayın:
        - soruyu ve metni tokenize edin
        - modeli çalıştır
        - en yüksek start ve end skorlarına sahip tokenleri bulun
        - cevabı oluşturun ve döndürün
    4. Bir metin ve soru tanımlayın
    5. predict_answer fonksiyonunu kullanarak cevabı bulun ve yazdırın

kurulumlar:
    pip install transformers torch
"""
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

model_name= "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer= BertTokenizer.from_pretrained(model_name) # modelin tokenizer'ını yükler, bu tokenizer metni modele uygun şekilde tokenize etmek için kullanılır

model= BertForQuestionAnswering.from_pretrained(model_name) # modelin kendisini yükler, bu model soru-cevap görevini gerçekleştirmek için eğitilmiştir

# soru cevaplama fonksiyonu
def predict_answer(question, context):
    # soru ve metni encode plus ile birleştir
    # return_tensors="pt" ifadesi, çıktının PyTorch tensor formatında olmasını sağlar
    # max_length=512 ifadesi, modelin işleyebileceği maksimum token sayısını belirtir, bu genellikle BERT modelleri için 512'dir
    # truncation=True ifadesi, eğer token sayısı maksimumu aşarsa, fazla tokenlerin kesilmesini sağlar

    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

    # input_ids, her kelimeyte karşılık gelen token kimlikleri
    input_ids = encoding["input_ids"]

    # attention_mask, modelin hangi tokenlere dikkat etmesi gerektiğini belirtir 
    attention_mask = encoding["attention_mask"]

    # modelin çıktısını al
    # start score: modelin her tokenin başlangıç pozisyonu için verdiği skor
    # end score: modelin her tokenin son pozisyonu için verdiği skor
    with torch.no_grad(): # modelin eğitilmediği ve sadece tahmin yapıldığı durumlarda, backpropagation yok
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False )

    # en yüksek start ve end skorlarına sahip tokenlerin indekslerini bul
    start_index = torch.argmax(start_scores, dim=1).item() 
    end_index = torch.argmax(end_scores, dim=1).item()

    # input_ids'ten start ve end indeksleri arasındaki tokenleri alarak cevabı oluştur
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])

    # tokenleri birleştirerek cevabı oluştur
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris, which is known for its art, culture, and history."

answer = predict_answer(question, context)
print(answer)
