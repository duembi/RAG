import chromadb
from chromadb.utils import embedding_functions
import ollama


system_prompt = """

Sen, Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı bünyesinde, OSB (Organize Sanayi Bölgesi) ve parsel verileri konusunda yardımcı olan bir asistansın. Kurumdaki OSB ve parsel uzmanı gibi bilgi veren, güvenilir bir bilirkişi rolündesin.

Kurallar:

1. Yalnızca sana verilen bağlam (context) belgelerinden faydalan. İnternetten bilgi çekme, kendi genel bilgine veya tahminlerine başvurma.

2. Kesinlikle halüsinasyon yapma (uydurma bilgi verme). Bu kurum Türkiye Cumhuriyeti Sanayi ve Teknoloji Bakanlığı'dır, verdiğin her bilgi kritik öneme sahiptir ve doğrulanabilir olmalıdır.

3. Verdiğin her cevabın, hangi bilgiye/kayda dayandığını (örneğin hangi OSB'nin sicil numarası, adı) mutlaka belirt — referans göstermeden cevap verme.

4. Eğer sorulan sorunun cevabı sana verilen bağlamda yoksa, şunu söyle: "Bu sorunun cevabı elimdeki belgelere göre verilemiyor." Eğer soru belirsizse veya birden fazla şekilde yorumlanabiliyorsa, kullanıcıya "Şunu mu sormak istediniz: ..." şeklinde açıklayıcı bir soru sor.

5. Kullanıcının sorduğu dilde cevap ver.

"""


chroma_client = chromadb.PersistentClient(path="./")

sentence_transformer_model = "distiluse-base-multilingual-cased-v1"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sentence_transformer_model)

chroma_collection = chroma_client.get_collection("rag_chunks", embedding_function=embedding_function)


def retrieveDocs(chroma_collection, query, n_results=5):
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
        output += f"Metin Parçası No: {i+1}\n{doc}\n"

    return output


messages = [
    {"role": "system", "content": system_prompt},
]

while True:

    user_input = input("Nasıl yardımcı olabilirim? (Çıkmak için Q yazınız): ")

    if user_input == "Q":
        print("Çıkış Yapılıyor..s")
        break

    result = retrieveDocs(chroma_collection, user_input)
    chunks = show_results(result)

    user_prompt = f"""
### Kullanıcı Sorusu:
{user_input}

### Erişilen Belgeler:
{chunks}
"""

    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(model="gemma3:4b", messages=messages)
    llm_answer = response.message.content

    print(f"\n{llm_answer}\n")

    messages.append({"role": "assistant", "content": llm_answer})
