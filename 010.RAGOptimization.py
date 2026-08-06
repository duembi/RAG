import chromadb
from chromadb.utils import embedding_functions
import ollama
import sqlite3
import datetime

system_prompt = """

Sen, Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı bünyesinde, OSB (Organize Sanayi Bölgesi) ve parsel verileri konusunda yardımcı olan bir asistansın. Kurumdaki OSB ve parsel uzmanı gibi bilgi veren, güvenilir bir bilirkişi rolündesin.

Kurallar:

1. Yalnızca sana verilen bağlam (context) belgelerinden faydalan. İnternetten bilgi çekme, kendi genel bilgine veya tahminlerine başvurma.

2. Kesinlikle halüsinasyon yapma (uydurma bilgi verme). Bu kurum Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'dır, verdiğin her bilgi kritik öneme sahiptir ve doğrulanabilir olmalıdır.

3. Verdiğin her cevabın, hangi bilgiye/kayda dayandığını (örneğin hangi OSB'nin sicil numarası, adı) mutlaka belirt — referans göstermeden cevap verme.

4. Eğer sorulan sorunun cevabı sana verilen bağlamda yoksa, şunu söyle: "Bu sorunun cevabı elimdeki belgelere göre verilemiyor." Eğer soru belirsizse veya birden fazla şekilde yorumlanabiliyorsa, kullanıcıya "Şunu mu sormak istediniz: ..." şeklinde açıklayıcı bir soru sor.

5. Kullanıcının sorduğu dilde cevap ver.

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

#DB bağlantısı oluşturuyorum
sqlitedb_client = sqlite3.connect("chatbot.db")
cursor = sqlitedb_client.cursor()


# 'users' tablosunu oluşturuyorum: her kullanıcıyı tek bir satırda temsil edecek
# id: her kullanıcıya otomatik atanan benzersiz numara
# username: kullanıcının adı (boş bırakılamaz, NOT NULL)
# date: kullanıcının oluşturulma tarihi
# IF NOT EXISTS sayesinde, bu kod tekrar çalıştırılsa bile tablo zaten varsa hata vermez, sessizce atlar
cursor.execute("""

    CREATE TABLE IF NOT EXISTS users (
    
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        date TEXT    

    )

""")
sqlitedb_client.commit() # değişikliği kalıcı hale getiriyorum, commit olmadan tablo diske yazılmış sayılmaz



# 'conversations' tablosunu oluşturuyorum: her bir sohbet oturumunu (konuşmayı) temsil edecek
# id: her konuşmaya otomatik atanan benzersiz numara
# user_id: bu konuşmanın hangi kullanıcıya ait olduğunu belirtir (users tablosundaki id'ye karşılık gelir)
# start_date: konuşmanın başladığı tarih
cursor.execute("""

    CREATE TABLE IF NOT EXISTS conversations(
    
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        start_date TEXT

    )

""")
sqlitedb_client.commit()



# 'messages' tablosunu oluşturuyorum: her tek mesajı (kullanıcı ya da asistan) temsil edecek
# id: her mesaja otomatik atanan benzersiz numara
# conversation_id: bu mesajın hangi konuşmaya ait olduğunu belirtir (conversations tablosundaki id'ye karşılık gelir)
# role: mesajı kimin gönderdiği ("system" / "user" / "assistant")
# content: mesajın asıl metni
# timestamp: mesajın gönderildiği zaman
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


chroma_client = chromadb.PersistentClient(path = "./")

sentence_transformer_model = "distiluse-base-multilingual-cased-v1"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sentence_transformer_model, device = "cuda" )

chroma_collection = chroma_client.get_collection("rag_chunks", embedding_function=embedding_function)


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
        print(metadatas[i])
        output += f"Metin Parçası No: {i+1}\nKaynak: {metadatas[i]['osb_adi']} (Sicil No: {metadatas[i]['sicil_no']}, Dönem: {metadatas[i]['donem']})\n{doc}\n"

    return output


messages = [

    {"role": "system", "content": system_prompt},

]




def format_history(messages):
    history_text = ""

    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            history_text = history_text+ f"Kullanıcı: {msg['content']}\n"    
        elif msg["role"] == "assistant":
            history_text = history_text+ f"Asistant: {msg['content']}\n"

    return history_text             


def generate_query_enhancement(user_input, messages):

    chat_history = format_history(messages)

    better_messages = [

        {"role" : "system", "content" : system_prompt_enhancement},
        {"role" : "user", "content" : f"Konuşma Geçmişi : {chat_history}"f"son_soru : {user_input}"}

    ]

    new_response = ollama.chat(model = "gemma3:4b", messages = better_messages)
    llm_ans = new_response.message.content            
    return llm_ans

def is_meta_question(user_input):

    mqmessages = [

    {"role" : "system", "content" : system_prompt_meta_check},
    {"role" : "user", "content" : user_input}

        
    ]

    ans = ollama.chat(model = "gemma3:4b", messages = mqmessages)

    if "EVET" in ans.message.content:
        return True
    else:
        return False

dt = str(datetime.datetime.now())
cursor.execute("INSERT INTO conversations (user_id, start_date) VALUES (?,?)", (1, dt))                                                      
sqlitedb_client.commit()
current_conversation_id = cursor.lastrowid


cursor.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)", (current_conversation_id, "system", system_prompt, f"{datetime.datetime.now()}"))
sqlitedb_client.commit()
 
while True:

    user_input = input("Nasıl yardımcı olabilirim? (Çıkmak için Q yazınız): ")

    if user_input == "Q" or user_input == "q":
        print("Çıkış Yapılıyor..s")
        break
         

    if is_meta_question(user_input):
        chunks = "Bu soru için belge erişimi yapılmadı, konuşma geçmişine bakınız."

    else:    
        enhanced_query = generate_query_enhancement(user_input, messages)

        if "?" in enhanced_query:
            result = retrieveDocs(chroma_collection, enhanced_query)
        else:
            result = retrieveDocs(chroma_collection, user_input)      
            

        print(enhanced_query)
        chunks = show_results(result)

    user_prompt = f"""

### Kullanıcı Sorusu:
{user_input}

### Erişilen Belgeler:
{chunks}
"""

    messages.append({"role": "user", "content": user_prompt})
    cursor.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)", (current_conversation_id, "user", user_input, f"{datetime.datetime.now()}"))
    sqlitedb_client.commit()




    response = ollama.chat(model="gemma3:4b", messages=messages)
    llm_answer = response.message.content

    print(f"Total Duration : {response.total_duration}")
    print(f"Prompt Eval Count : {response.prompt_eval_count}")
    print(f"\n{llm_answer}\n")

    messages.append({"role": "assistant", "content": llm_answer})
    cursor.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?,?,?,?)", (current_conversation_id, "assistant", llm_answer, f"{datetime.datetime.now()}"))
    sqlitedb_client.commit()


             
