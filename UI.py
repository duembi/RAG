import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import ollama
import sqlite3
import datetime

# =============================================================================
# SYSTEM PROMPT'LAR (senin orijinal kodundan aynen alındı)
# =============================================================================

system_prompt = """

Sen, Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı bünyesinde, OSB (Organize Sanayi Bölgesi) ve parsel verileri konusunda yardımcı olan bir asistansın. Kurumdaki OSB ve parsel uzmanı gibi bilgi veren, güvenilir bir bilirkişi rolündesin.

Bir soruyu cevaplarken şu SIRAYI takip et:
ADIM 1: Önce kural 2'deki İLGİLİLİK KONTROLÜNÜ yap. Bu kontrolü DİĞER HİÇBİR KURALDAN ÖNCE atlama.
ADIM 2: İlgililik kontrolünden geçen belgeler varsa, kural 3-7'deki diğer talimatları uygula.

Kurallar:

1. Yalnızca sana verilen bağlam (context) belgelerinden faydalan. İnternetten bilgi çekme, kendi genel bilgine veya tahminlerine başvurma.

2. İLGİLİLİK KONTROLÜ (HER ZAMAN İLK YAPILACAK ADIM): Sana verilen belge parçaları (context), bir arama sisteminden otomatik olarak geliyor ve bu sistem bazen soruyla alakasız belgeler getirebilir (örneğin isim/kelime benzerliği yüzünden — "Mars" ile "Mardin" gibi). Bu yüzden, context'te gördüğün her belgeyi otomatik olarak doğru eşleşme SAYMA. Diğer hiçbir kuralı uygulamadan ÖNCE, her belgenin sorudaki konu/isim/yer ile GERÇEKTEN ilgili olup olmadığını değerlendir. İlgili değilse o belgeyi tamamen yok say ve sanki context'te hiç yokmuş gibi davran — bu belgede geçen hiçbir sayıyı, ismi veya kaydı cevabına dahil etme. Belgelerin hiçbiri soruyla ilgili değilse, kural 5'teki gibi "Bu sorunun cevabı elimdeki belgelere göre verilemiyor" de ve DUR, başka hiçbir şey ekleme.

3. Kesinlikle halüsinasyon yapma (uydurma bilgi verme). Bu kurum Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'dır, verdiğin her bilgi kritik öneme sahiptir ve doğrulanabilir olmalıdır.

4. Verdiğin her cevabın, hangi bilgiye/kayda dayandığını (örneğin hangi OSB'nin sicil numarası, adı) mutlaka belirt — referans göstermeden cevap verme.

5. Eğer sorulan sorunun cevabı sana verilen bağlamda yoksa, şunu söyle: "Bu sorunun cevabı elimdeki belgelere göre verilemiyor." Eğer soru belirsizse veya birden fazla şekilde yorumlanabiliyorsa, kullanıcıya "Şunu mu sormak istediniz: ..." şeklinde açıklayıcı bir soru sor.

6. Kullanıcının sorduğu dilde cevap ver.

7. İlgililik kontrolünden (kural 2) geçen belge parçalarında iki farklı kayıt türü olabilir: ÖZET kayıtları ve SEKTÖR kayıtları (her parçanın başında bu bilgi açıkça belirtilir). Bunları KESİNLİKLE birbirine karıştırma:
   - "Toplam İstihdam" alanı, sadece ÖZET kayıtlarında bulunur ve o OSB'nin GERÇEK, GÜNCEL toplam istihdamını gösterir. Kullanıcı bir OSB'nin şu anki/mevcut/toplam istihdamını sorduğunda SADECE bu alanı kullan.
   - ÖZET kayıtlarında ayrıca "Öngörü İstihdam" adında BAŞKA bir alan daha bulunur — bu GELECEKTEKİ tahmini/hedef istihdamdır, GÜNCEL durumu YANSITMAZ. Kullanıcı açıkça "öngörü", "tahmini", "hedef" demedikçe bu alanı KULLANMA.
   - "istihdam" alanı SEKTÖR kayıtlarında bulunur, sadece o sektöre özeldir. Bir OSB'nin birden fazla sektör kaydının "istihdam" değerlerini toplayarak toplam istihdamı hesaplamaya ÇALIŞMA.
   - Aynı "osb_adi" değerine sahip olsalar bile, farklı "Sicil No" değerine sahip kayıtlar FARKLI OSB'lerdir; her Sicil No'yu ayrı bir bölge olarak ele al ve hangi sayının hangi Sicil No'ya ait olduğunu net şekilde belirt.
   Bu kural YALNIZCA sana gerçek belge parçaları verildiğinde (yani "Kaynak: ... Sicil No: ..." formatında somut kayıtlar gördüğünde) geçerlidir. Eğer "Erişilen Belgeler" bölümünde bunun yerine "Bu soru için belge erişimi yapılmadı, konuşma geçmişine bakınız" gibi bir metin görüyorsan, bu kuralı UYGULAMA. Böyle bir durumda belge aramıyorsun; bunun yerine bu konuşmadaki önceki kullanıcı mesajlarına ve senin verdiğin cevaplara bakarak soruyu cevapla.
"""

system_prompt_enhancement = """

Sen, Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'nın OSB ve parsel veri sistemi için çalışan bir sorgu iyileştirme uzmanısın. Görevin, kullanıcıdan gelen soruyu CEVAPLAMAK DEĞİL, o soruyu vektör veritabanında arama yapmaya uygun, kendi başına anlamlı bir ARAMA SORGUSUNA dönüştürmek.

Sana iki bilgi verilecek: konuşma geçmişi (chat_history) ve kullanıcının son sorusu (human_query).

KESİN KURALLAR:

1. ASLA cevap üretme. Elindeki bilgilerden hareketle sayı, isim, sektör kodu gibi somut veriler UYDURMA veya TAHMİN ETME. Senin görevin bilgi vermek değil, soruyu yeniden yazmak.

2. Kullanıcı, teknik terminolojiye her zaman hakim olmayabilir. Günlük ifadeleri, veri setindeki karşılıklarına yaklaştır (örneğin "boş arsa" yerine "boş parsel", "dolu parsel" yerine "tahsisli parsel" gibi).

3. Son sorudaki zamirleri ve eksik referansları ("bu", "o", "orası", "peki", "bunlar" gibi) konuşma geçmişine bakarak çöz. Son soruyu, geçmişten tamamen bağımsız olarak okunduğunda bile anlaşılır olacak şekilde yeniden yaz.

4. Eğer son soru zaten kendi başına anlamlıysa ve geçmişe ihtiyaç duymuyorsa, olduğu gibi (gerekirse sadece terminoloji düzeltmesiyle) bırak.

5. Çıktın KESİNLİKLE tek bir SORU CÜMLESİ olmalı ("kaç", "hangi", "nedir", "kimdir" gibi soru ifadeleriyle bitmeli veya soru işareti içermeli). Açıklama, yorum, giriş cümlesi, gerekçe EKLEME. Sadece soruyu döndür.

Örnekler:

Geçmiş: "Kullanıcı: Kayseri OSB (Sicil No: 416) hakkında bilgi ver."
Son Soru: "orada tahsisli parsel sayısı kaç?"
Doğru Çıktı: "Kayseri OSB'nin (Sicil No: 416) tahsisli parsel sayısı kaçtır?"
Yanlış Çıktı: "Kayseri OSB'de (Sicil No: 416) 5 adet tahsisli parsel bulunmaktadır." (Bu bir CEVAP, YASAK.)

Geçmiş: "Kullanıcı: Kayseri'deki OSB'ler hangileri?"
Son Soru: "bunların en büyüğü hangisi"
Doğru Çıktı: "Kayseri'deki OSB'ler arasında en büyük olanı hangisidir?"

"""

system_prompt_meta_check = """

    Sen, bir sınıflandırma asistanısın. Görevin, kullanıcıdan gelen bir sorunun İKİ KATEGORİDEN hangisine ait olduğuna karar vermek. Cevap ÜRETMEYECEKSİN, sadece sınıflandırma yapacaksın.

    KATEGORİLER:

    1. VERİ SORUSU: Soru, OSB (Organize Sanayi Bölgesi), parsel, sicil no, sektör, istihdam, fabrika sayısı gibi somut bilgi/veri talep ediyor. Bu sorular vektör veritabanından belge aramayı gerektirir.
    Örnekler: "Kayseri'de tekstil sektöründe faaliyet gösteren OSB'ler hangileri?", "orada kaç fabrika var?", "en çok boş parseli olan OSB hangisi?"

    2. META-SORU: Soru, OSB/parsel verisiyle değil, bu SOHBETİN KENDİSİYLE ilgilidir — yani kullanıcı, daha önce ne sorduğunu, sana neyi konuştuğunuzu, önceki mesajları hatırlamanı istiyor. Bu sorular belge aramayı GEREKTİRMEZ, çünkü cevap zaten konuşma geçmişinde mevcuttur.
    Örnekler: "Az önce sana ne sordum?", "daha önce hangi OSB'den bahsettik?", "ilk sorum neydi?", "bu konuşmada neler konuştuk?", "önceki cevabını tekrar eder misin?"

    KESİN KURALLAR:

    1. Cevabın SADECE şu iki kelimeden biri olmalı: "EVET" (eğer META-SORU ise) veya "HAYIR" (eğer VERİ SORUSU ise).

    2. Başka HİÇBİR açıklama, yorum, noktalama işareti veya ek kelime YAZMA. Çıktın tam olarak "EVET" ya da "HAYIR" olmalı, başka bir şey değil.

    3. Emin olamadığın durumlarda (soru hem veri hem geçmişle ilgili gibi görünüyorsa, örneğin "orada kaç fabrika var?" gibi zamir içeren ama veri talep eden sorular), "HAYIR" (VERİ SORUSU) olarak sınıflandır — çünkü bu tür sorular aslında zamir çözümlemesiyle (sorgu iyileştirme adımıyla) ele alınıyor, meta-soru değildir.

    Örnekler:

    Soru: "Az önce sana hangi soruları sordum?"
    Çıktı: EVET

    Soru: "Kayseri'de tekstil sektöründe faaliyet gösteren OSB'ler hangileri?"
    Çıktı: HAYIR

    Soru: "Orada kaç fabrika var?"
    Çıktı: HAYIR

    Soru: "Biraz önce konuştuğumuz OSB'nin adı neydi?"
    Çıktı: EVET

"""

# =============================================================================
# DB KURULUMU (senin orijinal kodundan aynen alındı, check_same_thread eklendi)
# =============================================================================

sqlitedb_client = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = sqlitedb_client.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        date TEXT
    )
""")
sqlitedb_client.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        start_date TEXT
    )
""")
sqlitedb_client.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )
""")
sqlitedb_client.commit()

# =============================================================================
# CHROMA / RETRIEVAL (senin orijinal kodundan aynen alındı)
# =============================================================================

chroma_client = chromadb.PersistentClient(path="./")
sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=sentence_transformer_model, device="cuda"
)
chroma_collection = chroma_client.get_collection("rag_chunks", embedding_function=embedding_function)
DISTANCE_THRISHOLD = 0.75


def retrieveDocs(chroma_collection, query, n_results=10):
    ans = chroma_collection.query(
        query_texts=[query],
        include=["documents", "metadatas", "distances"],
        n_results=n_results,
    )
    return ans


def show_results(ans):
    output = ""
    documents = ans['documents'][0]
    metadatas = ans['metadatas'][0]
    distances = ans['distances'][0]

    for i, doc in enumerate(documents):
        if distances[i] > DISTANCE_THRISHOLD:
            continue

        if metadatas[i].get('tur') == "osb_ozet":
            tur_aciklamasi = "Bu bir ÖZET kaydıdır. Bu OSB'nin genel/toplam istihdamı 'Toplam İstihdam' alanındadır."
        elif metadatas[i].get('tur') == 'osb_sektor':
            tur_aciklamasi = "Bu bir SEKTÖR kaydıdır. Buradaki 'istihdam' değeri sadece bu sektöre özeldir, OSB'nin toplam istihdamı DEĞİLDİR, toplama katma."
        else:
            tur_aciklamasi = ""

        output += f"Metin Parçası No: {i+1}\nKaynak: {metadatas[i]['osb_adi']} (Sicil No: {metadatas[i]['sicil_no']}, Dönem: {metadatas[i]['donem']})\n{tur_aciklamasi}\n{doc}\n"

    if output == "":
        output = "Sorguyla yeterince ilgili bir belge bulunamadı !!"

    return output


def format_history(messages):
    history_text = ""
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            history_text += f"Kullanıcı: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"Asistant: {msg['content']}\n"
    return history_text


def generate_query_enhancement(user_input, messages):
    chat_history = format_history(messages)

    better_messages = [
        {"role": "system", "content": system_prompt_enhancement},
        {"role": "user", "content": f"Konuşma Geçmişi : {chat_history}\nSon Soru : {user_input}"},
    ]

    new_response = ollama.chat(model="gemma3:4b", messages=better_messages)
    return new_response.message.content


def is_meta_question(user_input):
    mqmessages = [
        {"role": "system", "content": system_prompt_meta_check},
        {"role": "user", "content": user_input},
    ]
    ans = ollama.chat(model="gemma3:4b", messages=mqmessages)
    return "EVET" in ans.message.content


def extract_user_question(content):
    """DB'de saklanan tam user_prompt bloğundan, Chatbot'ta göstermek için
    sadece kullanıcının yazdığı temiz soruyu ayıklar."""
    if "### Kullanıcı Sorusu:" in content:
        try:
            after = content.split("### Kullanıcı Sorusu:")[1]
            question = after.split("### Erişilen Belgeler:")[0]
            return question.strip()
        except Exception:
            return content
    return content


# =============================================================================
# GRADIO ARAYÜZ FONKSİYONLARI
# =============================================================================

def yeni_konusma_baslat():
    dt = str(datetime.datetime.now())
    cursor.execute("INSERT INTO conversations (user_id, start_date) VALUES (?,?)", (1, dt))
    sqlitedb_client.commit()
    current_conversation_id = cursor.lastrowid

    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)",
        (current_conversation_id, "system", system_prompt, str(datetime.datetime.now())),
    )
    sqlitedb_client.commit()

    yeni_messages = [{"role": "system", "content": system_prompt}]

    return (
        gr.update(visible=False),   # group_start
        gr.update(visible=True),    # group_chat
        current_conversation_id,    # state_conv_id
        yeni_messages,               # state_messages
        [],                          # chatbot (ekranda gösterilecek liste, boş başlıyor)
    )


def gecmis_konusmalari_getir():
    cursor.execute("SELECT id, start_date FROM conversations ORDER BY id;")
    kayitli_konusmalar = cursor.fetchall()

    if len(kayitli_konusmalar) == 0:
        return gr.update(choices=[], value=None, visible=True), gr.update(visible=True)

    secenekler = [(f"ID: {row[0]} - {row[1]}", row[0]) for row in kayitli_konusmalar]
    return gr.update(choices=secenekler, value=None, visible=True), gr.update(visible=True)


def devam_et_baslat(secilen_id):
    if secilen_id is None:
        # Hiçbir şey seçilmeden onaylanırsa ekranı değiştirme
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            [],
            [],
        )

    current_conversation_id = int(secilen_id)

    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id;",
        (current_conversation_id,),
    )
    rows = cursor.fetchall()

    yeni_messages = []
    chatbot_display = []
    bekleyen_soru = None

    for role, content in rows:
        yeni_messages.append({"role": role, "content": content})

        if role == "user":
            bekleyen_soru = extract_user_question(content)
        elif role == "assistant" and bekleyen_soru is not None:
            chatbot_display.append({"role": "user", "content": bekleyen_soru})
            chatbot_display.append({"role": "assistant", "content": content})
            bekleyen_soru = None

    return (
        gr.update(visible=False),   # group_start
        gr.update(visible=True),    # group_chat
        current_conversation_id,    # state_conv_id
        yeni_messages,               # state_messages
        chatbot_display,             # chatbot
    )


def mesaj_gonder(user_input, state_messages, state_conv_id, chatbot_display):
    if not user_input or not user_input.strip():
        return chatbot_display, state_messages, state_conv_id, ""

    if is_meta_question(user_input):
        chunks = "Bu soru için belge erişimi yapılmadı, konuşma geçmişine bakınız."
    else:
        enhanced_query = generate_query_enhancement(user_input, state_messages)

        if "?" in enhanced_query:
            result = retrieveDocs(chroma_collection, enhanced_query)
        else:
            result = retrieveDocs(chroma_collection, user_input)

        chunks = show_results(result)

    user_prompt = f"""

### Kullanıcı Sorusu:
{user_input}

### Erişilen Belgeler:
{chunks}
"""

    state_messages = state_messages + [{"role": "user", "content": user_prompt}]
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)",
        (state_conv_id, "user", user_prompt, str(datetime.datetime.now())),
    )
    sqlitedb_client.commit()

    response = ollama.chat(model="gemma3:4b", messages=state_messages)
    llm_answer = response.message.content

    state_messages = state_messages + [{"role": "assistant", "content": llm_answer}]
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)",
        (state_conv_id, "assistant", llm_answer, str(datetime.datetime.now())),
    )
    sqlitedb_client.commit()

    chatbot_display = chatbot_display + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": llm_answer},
    ]

    return chatbot_display, state_messages, state_conv_id, ""


# =============================================================================
# ARAYÜZ (gr.Blocks)
#
# Tasarım konsepti: "kadastro sicili / imar paftası" — OSB ve parsel verisiyle
# çalışan bir aracın dünyasından ödünç alınmış bir dil: fon üzerinde çok soluk
# bir imar paftası ızgarası, kartların köşelerinde teknik çizim tescil işaretleri
# (registration marks), veri/kimlik alanları için mono yazı tipi, resmi bir
# evrak başlığı gibi kurgulanmış üst bant. Renk: "Blueprint Blue" birincil aksan,
# "Registry Amber" ikincil/onay aksanı (resmi bir kaşe/damga rengi gibi).
# =============================================================================

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#FAFAF8",
    background_fill_primary="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E5EA",
    block_border_width="1px",
    block_shadow="none",
    block_radius="2px",
    button_primary_background_fill="#2B4C7E",
    button_primary_background_fill_hover="#1F3860",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F3F1EC",
    button_secondary_border_color="#C77D2E",
    button_secondary_text_color="#9A5A1E",
    button_large_radius="2px",
    button_small_radius="2px",
    input_radius="2px",
    input_border_color="#D7DAE0",
    body_text_color="#16233F",
    body_text_color_subdued="#5B6472",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #16233F;
    --blueprint: #2B4C7E;
    --amber: #C77D2E;
    --slate: #5B6472;
    --grid-line: #E2E5EA;
    --paper: #FAFAF8;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    background:
        repeating-linear-gradient(0deg, rgba(43,76,126,0.05) 0 1px, transparent 1px 32px),
        repeating-linear-gradient(90deg, rgba(43,76,126,0.05) 0 1px, transparent 1px 32px),
        var(--paper) !important;
}

/* ---- Evrak başlığı (letterhead) ---- */
#app-header {
    padding: 22px 4px 16px 4px;
    border-bottom: 3px double var(--ink);
    margin-bottom: 22px;
}
#app-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--slate);
}
#app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: var(--ink);
    margin: 4px 0 2px 0;
    letter-spacing: -0.01em;
}
#app-sub {
    color: var(--slate);
    font-size: 0.92rem;
    margin: 0;
}
#app-refno {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--amber);
    margin-top: 6px;
}

/* ---- Teknik çizim köşe tescil işaretleri (signature element) ---- */
.reg-card {
    position: relative;
    padding: 30px 26px !important;
}
.reg-card::before, .reg-card::after,
.reg-card .corner-br::before, .reg-card .corner-br::after {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    border: 2px solid var(--blueprint);
    opacity: 0.55;
}
.reg-card::before { top: 6px; left: 6px; border-right: none; border-bottom: none; }
.reg-card::after { top: 6px; right: 6px; border-left: none; border-bottom: none; }

#chat-card { padding: 14px !important; }
#chat-card::before, #chat-card::after { opacity: 0.35; }

/* Kart içindeki inputlar Gradio'nun kendi kutu stilini (kenarlık+gölge) miras
   alıp "kart içinde kart" görüntüsü oluşturmasın diye, bu bileşenlere verilen
   elem_classes="flat-field" üzerinden çerçeveleri kaldırılıyor — dış .reg-card
   zaten tek çerçeveyi veriyor. (Not: Gradio'nun iç class isimleri sürümden
   sürüme değişiyor/hash'leniyor, o yüzden hedefleme sadece elem_classes ile
   yapılıyor — bu Gradio'nun CSS için desteklediği kararlı yol.) */
.flat-field {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* ---- Butonlar & girişler ---- */
#send-row { align-items: flex-end; }
.gr-button { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; }

/* ---- Alt bilgi şeridi (döküman kimliği gibi) ---- */
#doc-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--slate);
    text-align: center;
    letter-spacing: 0.06em;
    padding: 18px 0 6px 0;
    opacity: 0.7;
}

/* ---- Erişilebilirlik: klavye odağı görünür kalsın ---- */
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 2px solid var(--amber) !important;
    outline-offset: 2px;
}

footer { visibility: hidden }
"""


with gr.Blocks(title="STB OSB/Parsel Asistanı") as demo:

    state_messages = gr.State([])
    state_conv_id = gr.State(None)

    gr.HTML(
        """
        <div id="app-header">
            <div id="app-eyebrow">T.C. Sanayi ve Teknoloji Bakanlığı — OSB Genel Müdürlüğü</div>
            <div id="app-title">OSB / Parsel Sicil Asistanı</div>
            <p id="app-sub">Organize sanayi bölgesi ve parsel kayıtları üzerinden, yalnızca elindeki belgelere dayanarak yanıt veren bir sorgu aracı.</p>
            <div id="app-refno">SİSTEM: RAG-OSB · MODEL: gemma3:4b (yerel)</div>
        </div>
        """
    )

    with gr.Group(visible=True, elem_id="start-card", elem_classes=["reg-card"]) as group_start:
        gr.Markdown("### Nasıl başlamak istersiniz?")
        with gr.Row():
            btn_new = gr.Button("Yeni Kayıt Başlat", variant="primary", size="lg")
            btn_continue_toggle = gr.Button("Var Olan Kayıttan Devam Et", variant="secondary", size="lg")

        dropdown_conversations = gr.Dropdown(
            label="Kayıt No seç (Konuşma ID)", choices=[], visible=False,
            elem_classes=["flat-field"],
        )
        btn_continue_confirm = gr.Button("Seç ve Başla", visible=False, variant="primary")

    with gr.Group(visible=False, elem_id="chat-card", elem_classes=["reg-card"]) as group_chat:
        chatbot = gr.Chatbot(
            height=560,
            label=None,
            show_label=False,
            avatar_images=(None, "🏭"),
        )
        with gr.Row(elem_id="send-row"):
            textbox = gr.Textbox(
                placeholder="Örn: Kayseri OSB'nin toplam istihdamı kaçtır?",
                show_label=False,
                scale=8,
                autofocus=True,
            )
            btn_send = gr.Button("Gönder", variant="primary", scale=1, min_width=100)

    gr.HTML('<div id="doc-footer">STB · OSB-PARSEL VERİ SİSTEMİ · YALNIZCA İÇ KULLANIM İÇİN GELİŞTİRME AŞAMASI</div>')

    # --- Event bağlantıları ---

    btn_new.click(
        fn=yeni_konusma_baslat,
        outputs=[group_start, group_chat, state_conv_id, state_messages, chatbot],
    )

    btn_continue_toggle.click(
        fn=gecmis_konusmalari_getir,
        outputs=[dropdown_conversations, btn_continue_confirm],
    )

    btn_continue_confirm.click(
        fn=devam_et_baslat,
        inputs=[dropdown_conversations],
        outputs=[group_start, group_chat, state_conv_id, state_messages, chatbot],
    )

    btn_send.click(
        fn=mesaj_gonder,
        inputs=[textbox, state_messages, state_conv_id, chatbot],
        outputs=[chatbot, state_messages, state_conv_id, textbox],
    )

    textbox.submit(
        fn=mesaj_gonder,
        inputs=[textbox, state_messages, state_conv_id, chatbot],
        outputs=[chatbot, state_messages, state_conv_id, textbox],
    )


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
