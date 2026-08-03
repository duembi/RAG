import chromadb
from chromadb.utils import embedding_functions
import ollama

#diskteki kalıcı chromaDB veritabanına bağlanan bir clinet nesnesi oluşturur
chroma_client = chromadb.PersistentClient(path = "./") #diskteki kalıcı chromaDB veritabanına bağlanan bir clinet nesnesi oluşturur

sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
#metinleri vektöre çevireceğimiz fonskiyonu burada oluşuturuyoruz. Hem veri eklerken hem de şimdi sorgu yaparken aynı modeli kullanıyoruz burası önemli !!
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = sentence_transformer_model)
#diskte var olan rag_chunks collectionına bağlanıp, gelen sorguları vektöre çevirirken kullanacağı modeli ayarlıyoruz
chroma_collection = chroma_client.get_collection("rag_chunks", embedding_function = embedding_function)

system_prompt = """

Sen, Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı bünyesinde, OSB (Organize Sanayi Bölgesi) ve parsel verileri konusunda yardımcı olan bir asistansın. Kurumdaki OSB ve parsel uzmanı gibi bilgi veren, güvenilir bir bilirkişi rolündesin.

Kurallar:

1. Yalnızca sana verilen bağlam (context) belgelerinden faydalan. İnternetten bilgi çekme, kendi genel bilgine veya tahminlerine başvurma.

2. Kesinlikle halüsinasyon yapma (uydurma bilgi verme). Bu kurum Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'dır, verdiğin her bilgi kritik öneme sahiptir ve doğrulanabilir olmalıdır.

3. Verdiğin her cevabın, hangi bilgiye/kayda dayandığını (örneğin hangi OSB'nin sicil numarası, adı) mutlaka belirt — referans göstermeden cevap verme.

4. Eğer sorulan sorunun cevabı sana verilen bağlamda yoksa, şunu söyle: "Bu sorunun cevabı elimdeki belgelere göre verilemiyor." Eğer soru belirsizse veya birden fazla şekilde yorumlanabiliyorsa, kullanıcıya "Şunu mu sormak istediniz: ..." şeklinde açıklayıcı bir soru sor.

5. Kullanıcının sorduğu dilde cevap ver.

 """

#bu fonksiyon yaptığımız sorguya cevap getirecek ama bu sonuç raw olacak !!
def retrieveDocs(chroma_collection, query, n_results = 10) :
    ans = chroma_collection.query(query_texts = [query],
                            include= ["documents", "metadatas", "distances"],
                            n_results = n_results)
    
    return ans

query = "Kayseride tekstil sektörü olan osbler hangileri ?"
result = retrieveDocs(chroma_collection, query)
print(result)


#şimdi de gelen bu raw formatındaki cevabı okunabilir bir metne çevireceğiz 
def show_results(ans) : 

    output = ""
    documents = ans['documents'][0]
    metadatas = ans['metadatas'][0]
    distances = ans['distances'][0]

    for i ,doc in enumerate(documents) :
        output += f"Metin Parçasi No: {i+1}\n{doc}\n"

    return output

#print(show_results(sonuc))  

chunks = show_results(result)

user_prompt = f"""


###User Question
{query}

###STB Files
{chunks}

"""

print(user_prompt)

messages = [

    {"role" : "system", "content" : system_prompt},
    {"role" : "user", "content" : user_prompt}

]

response = ollama.chat(model = "gemma3:4b", messages = messages)
print(response.message.content)
