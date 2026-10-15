# pip install sentence-transformers scikit-learn
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def cumlelere_ayir(metin: str) -> list[str]:
    # Nokta, soru işareti, ünlem vb. noktalama işaretlerine göre cümleleri ayırır
    parcalar = re.split(r"(?<=[.!?…])\s+", metin.strip())
    return [c.strip() for c in parcalar if c.strip()]


def semantik_karsilastir(referans_metin: str, hedef_metin: str):
    # Modeli yerel olarak yükle
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Metinleri cümlelere böl
    ref_cumleler = cumlelere_ayir(referans_metin)
    hedef_cumleler = cumlelere_ayir(hedef_metin)

    if not ref_cumleler or not hedef_cumleler:
        print("Karşılaştırma yapılacak geçerli cümle bulunamadı.")
        return

    # Cümleleri embedding vektörlerine dönüştür
    ref_embeddings = model.encode(ref_cumleler)
    hedef_embeddings = model.encode(hedef_cumleler)

    # Cosine similarity hesapla (ref_cumleler x hedef_cumleler matrisi)
    benzerlik_matrisi = cosine_similarity(ref_embeddings, hedef_embeddings)

    print("=== Cümle Bazlı Semantik Karşılaştırma Sonuçları ===\n")
    for i, ref_cumle in enumerate(ref_cumleler):
        en_yuksek_skor_indeks = benzerlik_matrisi[i].argmax()
        en_yuksek_skor = benzerlik_matrisi[i][en_yuksek_skor_indeks]
        en_iyi_eslesen_cumle = hedef_cumleler[en_yuksek_skor_indeks]
        benzerlik_yuzdesi = en_yuksek_skor * 100

        print(f"Referans Cümle : {ref_cumle}")
        print(f"En Yakın Eşleşme: {en_iyi_eslesen_cumle}")
        print(f"Benzerlik Oranı : %{benzerlik_yuzdesi:.2f}")
        print("-" * 60)


if __name__ == "__main__":
    # Test Senaryosu
    ornek_referans = (
        "Yapay zeka teknolojileri modern dünyada hızla gelişmektedir. "
        "Yarın hava oldukça yağmurlu ve rüzgarlı olacak mı? "
        "Yazılım geliştiriciler algoritmaları optimize etmek için büyük çaba sarf eder!"
    )

    ornek_hedef = (
        "Meteoroloji raporlarına göre yarın şiddetli fırtına ve yağış bekleniyor. "
        "Programcılar kod performansını ve mantığını iyileştirmek adına sürekli çalışıyorlar. "
        "Makine öğrenimi ve yapay zeka alanındaki yenilikler günümüzde büyük bir ivme kazandı. "
        "Spor yapmak insan sağlığı için çok faydalıdır."
    )

    semantik_karsilastir(ornek_referans, ornek_hedef)
