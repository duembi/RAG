# STB OSB/Parcel RAG Chatbot

A fully local (self-hosted) RAG (Retrieval-Augmented Generation) chatbot built on top of the Turkish Ministry of Industry and Technology's Organized Industrial Zone (OSB) and parcel data. The system converts Excel-based OSB/parcel records into meaningful text chunks, stores them in a vector database, and answers user questions grounded strictly in that data, without hallucination.

This project follows the RAG architecture taught in Kasım Murat Karakaya's BTK Akademi RAG course, but replaces the Gemini API access layer with a **fully local `gemma3:4b` model served through Ollama**.

## Architecture

The system consists of two separate pipelines:

**Offline Data Pipeline** — a one-time preparation stage. Raw OSB/parcel Excel data (enriched with metadata such as `osb_adi`, `sicil_no`, `donem`) is parsed and split into meaningful text chunks, embedded on GPU using `distiluse-base-multilingual-cased-v1`, and persisted in ChromaDB.

**Online Inference Pipeline** — runs on every user query. It (1) classifies whether the question depends on chat history or is a self-contained data question, using Ollama/gemma3:4b (meta-question check); (2) if history-dependent, rewrites the query into a standalone question using the conversation history (query optimization); (3) embeds the query with the same embedding model; (4) performs a semantic similarity search in ChromaDB (distance threshold ≤ 0.75, distinguishing summary vs. sector records); (5) assembles the retrieved context together with the system rules into a single prompt; (6) generates the final answer via Ollama/gemma3:4b and returns it along with execution duration and prompt evaluation metrics.

Conversation state and message history are stored in SQLite (`chatbot.db`), while vectors live in a ChromaDB `PersistentClient` (collection `rag_chunks`).

<img width="600" alt="architecture" src="https://github.com/user-attachments/assets/f50d66f7-9fed-4cd8-a4b9-e48f1c47c8b8" />

## Project Stages

The code is organized as numbered files that follow the course sequence:

1. `001. First RAG Architect` — Introduction to RAG architecture
2. `002.Chatbot.py` — First basic chatbot using Ollama
3. `003.DataPreparation.py` — Converts the Excel data from wide to long format and turns it into meaningful Turkish sentences/chunks
4. `004. VectorDB.py` — ChromaDB client and collection setup
5. `005.LoadChunksToChroma.py` — Embeds the prepared chunks and loads them into ChromaDB
6. `006.Query.py` — First querying experiments against the vector database
7. `007. Context.py` — Building and formatting the retrieved context
8. `008.RAGChatbot.py` — Combines retrieval and generation into a single flow
9. `009.ImprovementChatbot:py` — Chatbot improvements
10. `010.RAGOptimization.py` — Query optimization, relevance filtering, summary/sector record separation, and other optimizations
11. `011.ConversationHistory.py` — Conversation/user/message history management with SQLite (console interface)
12. `UI.py` — Gradio-based web interface (final application)

## Tech Stack

- **LLM:** Local `gemma3:4b` via Ollama (no Gemini API)
- **Embedding model:** `distiluse-base-multilingual-cased-v1` (SentenceTransformers, GPU/CUDA)
- **Vector database:** ChromaDB (PersistentClient)
- **State/history storage:** SQLite (`chatbot.db`)
- **UI:** Gradio
- **Data processing:** pandas (Excel parsing)

## Setup

```bash
# Install Ollama and pull the model
ollama pull gemma3:4b

# Install Python dependencies
pip install chromadb sentence-transformers ollama gradio pandas openpyxl
```

**Hardware note:** `gemma3:4b` runs at the edge of feasibility on 4 GB VRAM GPUs (e.g. GTX 1650). Keeping the embedding model on the GPU at the same time can push VRAM usage over the limit; consider running the embedding model on `device="cpu"` or using a quantized model variant if needed.

## Usage

```bash
# 1) Convert the Excel data into chunks
python "003.DataPreparation.py"

# 2) Create the ChromaDB collection and load the chunks
python "005.LoadChunksToChroma.py"

# 3a) Chat from the console
python "011.ConversationHistory.py"

# 3b) or chat from the web UI
python UI.py
```

## Data Source

The input data is a periodic OSB/parcel table published by the Ministry of Industry and Technology (`Parsel_Tablosu.xlsx`). Each row represents one OSB; breakdowns for 24 different NACE sectors (NC, SA, PS, FS, İ) are packed side by side as columns. The data preparation step converts this "wide format" into a "long format" where each (OSB, sector) pair gets its own row, so that no information about which number belongs to which sector is lost during chunking/embedding.

## Limitations

This project is in a development/experimental stage and is intended for internal use only. The LLM is instructed to answer strictly based on the retrieved context documents; when no relevant context is found, it responds with "the answer to this question cannot be provided based on the available documents" instead of hallucinating.






# STB OSB/Parsel RAG Chatbot

Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'nın Organize Sanayi Bölgesi (OSB) ve parsel verileri üzerinde çalışan, tamamen yerel (self-hosted) bir RAG (Retrieval-Augmented Generation) sohbet botu. Sistem, Excel tabanlı OSB/parsel kayıtlarını anlamlı metin parçalarına dönüştürür, bunları bir vektör veritabanında saklar ve kullanıcı sorularını bu verilere dayanarak, halüsinasyon yapmadan cevaplar.

Bu proje, Kasım Murat Karakaya'nın BTK Akademi RAG eğitimindeki mimariyi temel alır; ancak LLM erişim katmanı Gemini API yerine **Ollama üzerinden yerel olarak çalışan `gemma3:4b`** modeline uyarlanmıştır.

## Mimari

Sistem iki ayrı hattan oluşur:

**Çevrimdışı Veri Hattı (Offline Data Pipeline)** — bir defalık hazırlık aşaması. Ham OSB/parsel Excel verisi (`osb_adi`, `sicil_no`, `donem` gibi meta verilerle birlikte) ayrıştırılıp anlamlı metin parçalarına (chunk) bölünür, `distiluse-base-multilingual-cased-v1` modeliyle GPU üzerinde vektörlere dönüştürülür ve ChromaDB'de kalıcı olarak saklanır.

**Çevrimiçi Çıkarım Hattı (Online Inference Pipeline)** — kullanıcı sorusu geldiğinde çalışır. Sırasıyla: (1) sorunun konuşma geçmişiyle mi ilgili olduğu yoksa bağımsız bir veri sorusu mu olduğu Ollama/gemma3:4b ile sınıflandırılır (meta-soru kontrolü); (2) geçmişe bağımlıysa soru, geçmiş konuşmaya göre kendi başına anlamlı hale getirilir (sorgu iyileştirme); (3) sorgu aynı embedding modeliyle vektöre çevrilir; (4) ChromaDB'de anlamsal benzerlik araması yapılır (mesafe eşiği ≤ 0.75, özet/sektör kayıtları ayrıştırılır); (5) bulunan bağlam, kurallarla birlikte bir sistem promptu içinde birleştirilir; (6) nihai cevap yine Ollama/gemma3:4b ile üretilir ve süre/metrik bilgileriyle birlikte kullanıcıya sunulur.

Konuşma geçmişi ve mesajlar SQLite (`chatbot.db`) içinde, vektörler ise ChromaDB `PersistentClient` (`rag_chunks` koleksiyonu) içinde saklanır.

<img width="600" alt="mimari" src="https://github.com/user-attachments/assets/8b4d2bf0-0190-4bb6-96f3-31493b25e938" />

## Proje Aşamaları

Kod, kurs sırasını takip eden numaralandırılmış dosyalar halinde ilerler:

1. `001. First RAG Architect` — RAG mimarisine giriş
2. `002.Chatbot.py` — Ollama ile ilk temel sohbet botu
3. `003.DataPreparation.py` — Excel (geniş format) verisini "dar format"a çevirip anlamlı Türkçe cümlelere/chunk'lara dönüştürme
4. `004. VectorDB.py` — ChromaDB istemcisi ve koleksiyon kurulumu
5. `005.LoadChunksToChroma.py` — Hazırlanan chunk'ları embedding'e çevirip ChromaDB'ye yükleme
6. `006.Query.py` — Vektör veritabanında ilk sorgu denemeleri
7. `007. Context.py` — Bağlam (context) oluşturma ve biçimlendirme
8. `008.RAGChatbot.py` — Retrieval + generation adımlarının birleştirilmesi
9. `009.ImprovementChatbot:py` — Sohbet botu iyileştirmeleri
10. `010.RAGOptimization.py` — Sorgu iyileştirme, alaka kontrolü, özet/sektör ayrımı gibi optimizasyonlar
11. `011.ConversationHistory.py` — SQLite ile kullanıcı/konuşma/mesaj geçmişi yönetimi (konsol arayüzü)
12. `UI.py` — Gradio tabanlı web arayüzü (nihai uygulama)

## Kullanılan Teknolojiler

- **LLM:** Ollama üzerinden yerel `gemma3:4b` (Gemini API kullanılmıyor)
- **Embedding modeli:** `distiluse-base-multilingual-cased-v1` (SentenceTransformers, GPU/CUDA)
- **Vektör veritabanı:** ChromaDB (PersistentClient)
- **Durum/geçmiş depolama:** SQLite (`chatbot.db`)
- **Arayüz:** Gradio
- **Veri işleme:** pandas (Excel ayrıştırma)

## Kurulum

```bash
# Ollama'yı kurun ve modeli indirin
ollama pull gemma3:4b

# Python bağımlılıklarını kurun
pip install chromadb sentence-transformers ollama gradio pandas openpyxl
```

**Donanım notu:** `gemma3:4b`, 4 GB VRAM'li kartlarda (örn. GTX 1650) sınırda çalışır. Embedding modelini de aynı anda GPU'da tutmak VRAM yetersizliğine yol açabilir; gerekirse embedding modelini `device="cpu"` ile çalıştırmayı veya quantize edilmiş model varyantlarını değerlendirin.

## Kullanım

```bash
# 1) Excel verisini chunk'lara dönüştür
python "003.DataPreparation.py"

# 2) ChromaDB koleksiyonunu oluştur ve chunk'ları yükle
python "005.LoadChunksToChroma.py"

# 3a) Konsoldan sohbet et
python "011.ConversationHistory.py"

# 3b) veya web arayüzünden sohbet et
python UI.py
```

## Veri Kaynağı

Girdi verisi, Sanayi ve Teknoloji Bakanlığı'na ait dönemsel OSB/parsel tablosudur (`Parsel_Tablosu.xlsx`). Her satır bir OSB'yi temsil eder; 24 farklı NACE sektörüne ait kırılımlar (NC, SA, PS, FS, İ) yan yana sütunlar halinde tutulur. Veri hazırlama adımında bu "geniş format", her (OSB, sektör) çiftini kendi satırına açan bir "dar format"a dönüştürülür; böylece hangi sayının hangi sektöre ait olduğu embedding/chunking sırasında kaybolmaz.

## Sınırlamalar

Bu proje geliştirme/deneme aşamasındadır ve yalnızca iç kullanım amaçlıdır. LLM yalnızca kendisine verilen bağlam belgelerine dayanarak cevap verecek şekilde yönlendirilmiştir; bağlamda karşılık bulamadığı sorularda halüsinasyon üretmek yerine "bu sorunun cevabı elimdeki belgelere göre verilemiyor" yanıtını döndürür.
