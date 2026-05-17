"""
Amaç: 

- Bu uygulamanın amacı, Hugging face MarianMT modeilini kullanarak çeviri yapmaktır. İngilizce - Fransızca çeviri gerçekleştirelim.
- Model, Helsinki-NLP tarafından eğitilmiş bir MarianMT modelidir. Çok sayıda dil çifti arasında çeviri yapmak üzere eğitildi.

Adımlar:
    1. Gerekli kütüphaneleri yükleyin
    2. model adını belirle
    3. model ve tokenizer'ı yükleyin
    4. Çeviri yapmak istediğiniz metni tanımlayın
    5. Metni tokenize edin ve modelin çeviri yapmasını sağlayın
    6. Çeviriyi yazdırın

kurulumlar:
    pip install transformers torch sentencepiece

"""

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr" # İngilizce - Fransızca çeviri modeli

tokenizer = MarianTokenizer.from_pretrained(model_name) # modelin tokenizer'ını yükler, bu tokenizer metni modele uygun şekilde tokenize etmek için kullanılır

model = MarianMTModel.from_pretrained(model_name) # modelin kendisini yükler

text="Selcan, I love you so much" # çeviri yapmak istediğimiz metin

# metni tokenize edin
inputs = tokenizer(text, return_tensors="pt", padding=True)

# model ile çeviri yapın
translated = model.generate(**inputs)

# skip_special_tokens=True ifadesi, çeviride özel tokenlerin atlanmasını sağlar --> <pad>, <eos>
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True) # çeviriyi decode ederek okunabilir hale getirir

print(f"Translated text: {translated_text}") # çeviriyi yazdırır