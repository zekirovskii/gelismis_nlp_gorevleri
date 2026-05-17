"""
Amaç: 
    - Hugginface transformers kütüphanesşnş kullanrak bir metin özetleme (summarization) modeli oluşturmak.
    - Verilen uzun metni kısa bir özet haline getirmek
    - modelimiz önceden eğitilmiş büyük bit dil modeli olsun, metin ana fikrini koruyarak ksıa bir özet üretir

Adımlar:
    1. Gerekli kütüphaneleri yükleyin
    2. özetleme summarization pipline'ını oluşturun
    3. Uzun bir metin tanımlayın
    4. Metni özetleyin
    5. Özetlenen metni yazdırın

kurumlar:
    pip install transformers torch
"""

from transformers import pipeline # transformers kütüphanesinden pipeline fonksiyonunu içe aktarır, önceden eğitilmiş modelleri kullanarak çeşitli NLP görevlerini gerçekleştirmek için kullanılır

summarizer=pipeline("summarization") 

# "summarization" görevini gerçekleştirmek için bir pipeline oluşturur, bu pipeline metin özetleme işlemi yapacak

# Uzun bir metin tanımlayın

text = """
It was a little past midnight in Istanbul. The last metro of the night was about to leave as Mert hurried down the wet streets toward the station. He had spent the entire day working in front of a computer screen, and exhaustion showed clearly on his face. Rain fell softly while the yellow lights of the street lamps reflected on the pavement.

When he entered the metro, the wagon was almost empty. He sat quietly near the window and reached for his phone, only to realize that the battery had died. As he looked around, he noticed an old man sitting across from him. The man held an old photograph in his hands, staring at it silently before taking deep breaths.

After a few minutes, the old man suddenly asked, “Do you think people can truly forget the past?”

Mert was surprised by the question. He had not expected a conversation with a stranger at this hour. After a short pause, he replied, “I don’t think so. Some memories stay with us forever.”

The old man smiled faintly and handed him the photograph. In the black-and-white picture, there was a young woman and a small child. “My daughter and grandson,” he said quietly. “I haven’t seen them for years.”

Mert listened carefully as the old man explained how a terrible argument had separated him from his family many years ago. Pride had stopped him from apologizing, and as time passed, reaching out became more difficult. Now, all he had left was an old photograph and years of regret.

As the metro crossed the bridge, the city lights reflected through the windows. Mert suddenly began thinking about his own life. He had not visited his family in a long time. Work had always seemed more important, and he constantly postponed calling them.

When the metro reached the old man’s stop, he slowly stood up. Before leaving, he turned to Mert and said, “People think they have endless time. But sometimes, they realize too late that they don’t.”

The doors closed behind him, and he disappeared into the crowd. Mert stared at his reflection in the dark window, feeling an unexpected sense of unease. For the first time in months, he understood how easily important relationships could fade away.

The moment he arrived home, he decided he would call his mother.
"""


summary = summarizer(
    text, max_length=80, # max 80 token
    min_length=5,# min 5 token
    do_sample=True # rasgele örnekleme yaparak farklı özetler üretmeye olanak tanır
) 

# summarier fonksiypnu bir liste return eder, her öğe bir sözlük yapısındadır.
print(summary[0]['summary_text']) # summary listesinin ilk öğesindeki 'summary_text' anahtarına erişerek özetlenen metni yazdırır


"""
max_lengt=20
 Mert had not visited his family in a long time; work seemed more important . 

 max_lengt=80
 Mert was surprised by an old man who asked him about the past . The man held an old photograph in his hands, staring at it silently before taking deep breaths . Mert listened carefully as the old man explained how a terrible argument had separated him from his family .


"""
