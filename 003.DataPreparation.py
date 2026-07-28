"""
STB OSB/Parsel Excel -> RAG dokumanlari

Amac:
  Excel'deki "genis format" (wide) yapiyi, RAG icin guvenli olan
  "dar format" (long) yapiya cevirmek ve her satiri anlamli bir
  Turkce cumleye (dokuman/chunk) donusturmek.

Neden gerekli:
  Excel'de her satir = 1 OSB. Ama her OSB satirinin icine, 24 farkli
  NACE sektorunun kirilimi YAN YANA SUTUNLAR olarak sikistirilmis:
      NC-1, SA-1, PS-1, FS-1, I-1 ... NC-24, SA-24, PS-24, FS-24, I-24
  Bu yapiyi duz metne cevirip karakter sayisina gore chunk'larsan,
  hangi sayinin hangi sektore ait oldugu bilgisi kaybolur.
  Cozum: her (OSB, sektor) ciftini KENDI satirina acmak.

Cikti dosyalari:
  osb_sektor_long.xlsx  -> dar format tablo (kontrol/inceleme icin)
  rag_chunks.jsonl      -> embedding'e verilecek dokumanlar + metadata
  rag_chunks_onizleme.txt -> ilk 30 dokumanin okunabilir hali
"""

import json
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# 0) Ayarlar
# --------------------------------------------------------------------------
EXCEL_YOLU = "2026-2__Çeyrek__Nisan-Mayıs-Haziran__Parsel_Tablosu (2).xlsx"
SAYFA = "Veri Tabanı"
CIKTI_KLASORU = Path(".")

DONEM = "2026 - 2. Çeyrek (Nisan-Mayıs-Haziran)"

# Tekrar eden sutun bloklarinin kokleri (stubnames)
# NOT: "İ" sutunu Turkce buyuk I. Kod icinde birebir bu karakter kullanilmali.
STUBLAR = ["NC", "SA", "PS", "FS", "İ"]

# Her OSB satirini benzersiz tanimlayan anahtar.
# DIKKAT: "OSB Adı" BENZERSIZ DEGIL (orn. "Afyon OSB" 9 kez geciyor).
# Gercek benzersiz anahtar "Sicil No".
ANAHTAR = "Sicil No"

# Dar formata acilirken her satirda tasinacak OSB duzeyindeki kimlik bilgileri
OSB_KIMLIK = [
    "ID", "Sicil No", "OSB Adı", "İl Adı", "İlçe", "Bölge",
    "OSB Türü", "Aşama", "Teşvik Bölgelerine Göre İller",
]

# OSB duzeyindeki ozet (toplam) sutunlari -> ayri bir dokuman turu olacak
OSB_OZET = [
    "Bölge Büyüklüğü (Ha)",
    "Sanayi Parsel Alanı (Ha) (x+y)",
    "Parsel Sayısı (İmar)",
    "Parsel Sayısı  (Bölge)",          # dikkat: iki bosluk var
    "Toplam Parsel Sayısı (Bölge ve Öngörü)",
    "Tahsisi Yapılan Parsellerin Sayısı (m)",
    "Tahsisi Yapılan Parsellerin Alanı (Ha) (x)",
    "Boş Parsel Sayısı (n)",
    "Boş Parsel Alan (Ha) (y)",
    "Üretim (a)",
    "İnşaat (b)",
    "Proje (c)",
    "Tahsisli Parsel Sayısı (a+b+c)",
    "Toplam İstihdam",      # gercek istihdam (bkz. asagidaki uyari)
    "Öngörü İstihdam",
    "OSB Kuruluş Yılı",
]

ASAMA_SOZLUK = {"İ": "İşletmede", "D": "Devam ediyor", "P": "Proje aşamasında"}


# --------------------------------------------------------------------------
# Yardimci fonksiyonlar
# --------------------------------------------------------------------------
def temizle(deger):
    """NaN -> None, string ise fazla bosluklari kirp."""
    if pd.isna(deger):
        return None
    if isinstance(deger, str):
        return re.sub(r"\s+", " ", deger).strip()
    return deger


def sayi_yaz(deger, birim=""):
    """Sayiyi okunabilir metne cevir. 6.0 -> '6', 1095.5 -> '1.095,5'."""
    if deger is None:
        return None
    if isinstance(deger, (int, float)):
        if float(deger).is_integer():
            metin = f"{int(deger):,}".replace(",", ".")
        else:
            metin = f"{deger:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
        return f"{metin} {birim}".strip()
    return str(deger)


# --------------------------------------------------------------------------
# 1) Excel'i oku
# --------------------------------------------------------------------------
df = pd.read_excel(EXCEL_YOLU, sheet_name=SAYFA)
df = df.dropna(how="all")                      # tamamen bos satirlari at
df = df[df["Sicil No"].notna()]                # sicil no olmayan satir gecersiz

print(f"[1] Excel okundu: {df.shape[0]} OSB kaydi, {df.shape[1]} sutun")

# Serbest not sutunu (Unnamed: 170) -> anlamli isim ver
if "Unnamed: 170" in df.columns:
    df = df.rename(columns={"Unnamed: 170": "Not"})

# --------------------------------------------------------------------------
# 2) GENIS -> DAR format (wide_to_long)
# --------------------------------------------------------------------------
# Sadece gerekli sutunlari alalim: kimlik + tekrar eden bloklar
blok_sutunlari = [f"{s}-{k}" for k in range(1, 25) for s in STUBLAR]
blok_sutunlari = [c for c in blok_sutunlari if c in df.columns]

alt = df[OSB_KIMLIK + blok_sutunlari].copy()

uzun = pd.wide_to_long(
    alt,
    stubnames=STUBLAR,   # NC, SA, PS, FS, İ
    i=ANAHTAR,           # her orijinal satiri tanimlayan anahtar
    j="sektor_slot",     # yeni olusacak indeks sutunu (1..24)
    sep="-",
    suffix=r"\d+",
).reset_index()

uzun = uzun.rename(columns={
    "NC": "nace_kodu",
    "SA": "sektor_adi",
    "PS": "parsel_sayisi",
    "FS": "fabrika_sayisi",
    "İ": "istihdam",
})

print(f"[2] Dar formata acildi: {len(uzun)} satir (= OSB x 24 sektor slotu)")

# --------------------------------------------------------------------------
# 3) Bos sektor slotlarini at
# --------------------------------------------------------------------------
# ONEMLI: NC ve SA sutunlari HER satirda dolu (sabit bir NACE sozlugu).
# Yani "NC bos mu" diye bakmak yanlis olur -> hicbir sey silinmez.
# Gercek dolulugun olcusu: PS / FS / İ degerlerinden en az biri dolu mu?
dolu_maske = uzun[["parsel_sayisi", "fabrika_sayisi", "istihdam"]].notna().any(axis=1)
uzun = uzun[dolu_maske].copy()

# Metin alanlarini temizle
uzun["sektor_adi"] = uzun["sektor_adi"].map(temizle)
for kolon in ["parsel_sayisi", "fabrika_sayisi", "istihdam"]:
    uzun[kolon] = pd.to_numeric(uzun[kolon], errors="coerce")

uzun = uzun.sort_values([ANAHTAR, "sektor_slot"]).reset_index(drop=True)
print(f"[3] Bos slotlar temizlendi: {len(uzun)} gercek (OSB x sektor) kaydi kaldi")

# --------------------------------------------------------------------------
# 4) SATIR -> DOKUMAN (row-to-document)
# --------------------------------------------------------------------------
dokumanlar = []

# 4a) Sektor bazli dokumanlar
for _, satir in uzun.iterrows():
    osb = temizle(satir["OSB Adı"])
    il = temizle(satir["İl Adı"])
    ilce = temizle(satir["İlçe"])
    sicil = temizle(satir[ANAHTAR])
    sektor = satir["sektor_adi"]
    nace = satir["nace_kodu"]

    # OSB'yi tekilleştiren tanim (ayni isimli OSB'ler var!)
    osb_tanim = f"{osb} (Sicil No: {sayi_yaz(sicil)}, {il}"
    osb_tanim += f" / {ilce})" if ilce else ")"

    parcalar = []
    if pd.notna(satir["parsel_sayisi"]):
        parcalar.append(f"{sayi_yaz(satir['parsel_sayisi'])} adet tahsisli parsel")
    if pd.notna(satir["fabrika_sayisi"]):
        parcalar.append(f"{sayi_yaz(satir['fabrika_sayisi'])} adet fabrika")
    if pd.notna(satir["istihdam"]):
        parcalar.append(f"{sayi_yaz(satir['istihdam'])} kişilik istihdam")

    metin = (
        f"{osb_tanim} bünyesinde, NACE {sayi_yaz(nace)} kodlu "
        f"\"{sektor}\" sektöründe {', '.join(parcalar)} bulunmaktadır. "
        f"Dönem: {DONEM}."
    )

    dokumanlar.append({
        "id": f"sektor-{int(sicil)}-{int(satir['sektor_slot'])}",
        "tur": "osb_sektor",
        "metin": metin,
        "meta": {
            "sicil_no": int(sicil),
            "osb_adi": osb,
            "il": il,
            "ilce": ilce,
            "bolge": temizle(satir["Bölge"]),
            "osb_turu": temizle(satir["OSB Türü"]),
            "nace_kodu": None if pd.isna(nace) else float(nace),
            "sektor_adi": sektor,
            "parsel_sayisi": None if pd.isna(satir["parsel_sayisi"]) else float(satir["parsel_sayisi"]),
            "fabrika_sayisi": None if pd.isna(satir["fabrika_sayisi"]) else float(satir["fabrika_sayisi"]),
            "istihdam": None if pd.isna(satir["istihdam"]) else float(satir["istihdam"]),
            "donem": DONEM,
        },
    })

# 4b) OSB ozet dokumanlari (sektor kirilimi olmayan, genel bilgiler)
for _, satir in df.iterrows():
    osb = temizle(satir["OSB Adı"])
    il = temizle(satir["İl Adı"])
    ilce = temizle(satir["İlçe"])
    sicil = temizle(satir[ANAHTAR])
    asama = ASAMA_SOZLUK.get(temizle(satir.get("Aşama")), temizle(satir.get("Aşama")))

    osb_tanim = f"{osb} (Sicil No: {sayi_yaz(sicil)}, {il}"
    osb_tanim += f" / {ilce})" if ilce else ")"

    cumleler = [
        f"{osb_tanim}, {temizle(satir.get('Bölge'))} bölgesinde yer alan "
        f"{temizle(satir.get('OSB Türü'))} tipi bir organize sanayi bölgesidir."
    ]
    if asama:
        cumleler.append(f"Aşama durumu: {asama}.")

    olcum = []
    for kolon, birim in [
        ("Bölge Büyüklüğü (Ha)", "hektar"),
        ("Sanayi Parsel Alanı (Ha) (x+y)", "hektar sanayi parsel alanı"),
        ("Parsel Sayısı  (Bölge)", "adet parsel (bölge)"),
        ("Tahsisi Yapılan Parsellerin Sayısı (m)", "adet tahsisli parsel"),
        ("Boş Parsel Sayısı (n)", "adet boş parsel"),
        ("Boş Parsel Alan (Ha) (y)", "hektar boş parsel alanı"),
        ("Üretim (a)", "adet üretimdeki parsel"),
        ("İnşaat (b)", "adet inşaat halindeki parsel"),
        ("Proje (c)", "adet proje aşamasındaki parsel"),
        # DIKKAT: Excel'deki "İstihdam" sutunu YANLIS ETIKETLI.
        # 500/500 satirda "Tahsisi Yapılan Parsellerin Sayısı (m)" ile birebir ayni.
        # Gercek istihdam rakami "Toplam İstihdam" sutunudur (sektor kirilimlarinin toplami).
        ("Toplam İstihdam", "kişi toplam istihdam"),
        ("Öngörü İstihdam", "kişi öngörü istihdam"),
    ]:
        deger = temizle(satir.get(kolon))
        if deger is not None:
            olcum.append(f"{sayi_yaz(deger, birim)}")

    if olcum:
        cumleler.append("Bölgede " + "; ".join(olcum) + " bulunmaktadır.")

    not_metni = temizle(satir.get("Not"))
    if not_metni:
        cumleler.append(f"Not: {not_metni}")

    cumleler.append(f"Dönem: {DONEM}.")

    dokumanlar.append({
        "id": f"osb-{int(sicil)}",
        "tur": "osb_ozet",
        "metin": " ".join(cumleler),
        "meta": {
            "sicil_no": int(sicil),
            "osb_adi": osb,
            "il": il,
            "ilce": ilce,
            "bolge": temizle(satir.get("Bölge")),
            "osb_turu": temizle(satir.get("OSB Türü")),
            "asama": asama,
            "donem": DONEM,
            **{
                k: (None if pd.isna(satir.get(k)) else float(satir.get(k)))
                for k in OSB_OZET
                if k in df.columns and pd.api.types.is_numeric_dtype(type(satir.get(k)))
            },
        },
    })

print(f"[4] Dokuman uretildi: {len(dokumanlar)} adet "
      f"({sum(d['tur'] == 'osb_sektor' for d in dokumanlar)} sektör + "
      f"{sum(d['tur'] == 'osb_ozet' for d in dokumanlar)} OSB özeti)")

# --------------------------------------------------------------------------
# 5) Kaydet
# --------------------------------------------------------------------------
uzun.to_excel(CIKTI_KLASORU / "osb_sektor_long.xlsx", index=False)

with open(CIKTI_KLASORU / "rag_chunks.jsonl", "w", encoding="utf-8") as f:
    for d in dokumanlar:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

with open(CIKTI_KLASORU / "rag_chunks_onizleme.txt", "w", encoding="utf-8") as f:
    for d in dokumanlar[:30]:
        f.write(f"--- {d['id']} [{d['tur']}] ---\n{d['metin']}\n\n")

print("[5] Kaydedildi: osb_sektor_long.xlsx, rag_chunks.jsonl, rag_chunks_onizleme.txt")
