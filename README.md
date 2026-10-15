# Semantic Similarity Checker

Bir referans metni ile karşılaştırılan metni **cümle bazında anlamsal (semantik)** olarak karşılaştıran Python aracı.

## Nasıl Çalışır?

1. Metinler regex ile (`.` `!` `?` `…`) cümlelere ayrılır.
2. Türkçe uyumlu `paraphrase-multilingual-MiniLM-L12-v2` modeli (sentence-transformers) ile cümleler embedding vektörlerine dönüştürülür.
3. `scikit-learn` `cosine_similarity` ile referans metindeki her cümlenin diğer metindeki cümlelerle benzerliği hesaplanır.
4. Her referans cümle için en yüksek eşleşen cümle ve benzerlik yüzdesi yazdırılır.

> Model lokalde çalışır, dış API kullanılmaz. Model ağırlıkları ilk çalıştırmada Hugging Face'ten bir kez indirilip önbelleğe alınır.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python semantic_similarity.py
```

## Örnek Çıktı

```
=== Cümle Bazlı Semantik Karşılaştırma Sonuçları ===

Referans Cümle : Yapay zeka teknolojileri modern dünyada hızla gelişmektedir.
En Yakın Eşleşme: Makine öğrenimi ve yapay zeka alanındaki yenilikler günümüzde büyük bir ivme kazandı.
Benzerlik Oranı : %84.10
------------------------------------------------------------
Referans Cümle : Yarın hava oldukça yağmurlu ve rüzgarlı olacak mı?
En Yakın Eşleşme: Meteoroloji raporlarına göre yarın şiddetli fırtına ve yağış bekleniyor.
Benzerlik Oranı : %64.68
------------------------------------------------------------
Referans Cümle : Yazılım geliştiriciler algoritmaları optimize etmek için büyük çaba sarf eder!
En Yakın Eşleşme: Programcılar kod performansını ve mantığını iyileştirmek adına sürekli çalışıyorlar.
Benzerlik Oranı : %65.77
------------------------------------------------------------
```
