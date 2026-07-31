
#chromaDB yi iki farklı şekilde çalıştırabilirz, biz kalıcı olan vektör DB kullanıcaz.
#Koleksiyonumuzu tanımlayıp, onu embdeding modeline vereceğiz
#otomatik embedding kurulacak
#metadata hazırlanacak, vereceğimi zverinin hangi kaynaktan geldiği bilgisni verir bize

import json
import chromadb
from chromadb.utils import embedding_functions

#vektör veritabnı için dosya yolu 
vector_database_path = "./"

#sentenceTransformer modelini kullanarak bir embedding fonksiyonu oluştur
#bu fonksiyon, metin parlaçarlarını vektöre dönüştürmek için kullanaılacak

#embedding modeli seçmi, sorgular ve metin parçaları bu modelden geçip vektörlere çevirilecek. Max 128 token!!
sentence_transformer_model = "distiluse-base-multilingual-cased-v1"

#fonskiyon oluşturma
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = sentence_transformer_model)

#persistent bir chromaDB istemcisi/clientı oluşturacağız
#bu sayede vektör veritabanının diske kaydedilmesini ve oturumlar arasında kalıcı olmasını sağlar
def create_chroma_client(vector_database_path, collection_name, embedding_function):
    chroma_client = chromadb.PersistentClient(path = vector_database_path)

    #burada bir collection oluşturuyoruz ama bu isiimde başka bir collection olabilir onun kontrolünü yapıyorum
    try :
        chroma_collection = chroma_client.get_collection(collection_name)

        #eğer bu isimde bir koleksiyon mevcutsa sil
        print(f"'{collection_name}' name existing. Deleting..")
        chroma_client.delete_collection(collection_name)
        print(f"'{collection_name}' deleted..")

        #eğer sildiysen silme işleminden sonra collection ı yeniden oluştur
        chroma_collection = chroma_client.create_collection(collection_name, embedding_function = embedding_function)
        print(f"'collection_name' collection created again..")

    except :
        #eğer koleksiyon mevcut değilse oluştur
        chroma_collection = chroma_client.create_collection(collection_name, embedding_function = embedding_function)
        print(f"'{collection_name}' created..")

#oluşturulan client ve collection ı döndür
    return chroma_client, chroma_collection    

collection_name = "rag_chunks"
chroma_client, chroma_collection = create_chroma_client(vector_database_path, collection_name, embedding_function)

print("Collection created !!")

print(f"\tCollection is in ChromaDB : {chroma_client.list_collections()}")
print(f"Total file number : {chroma_collection.count()}")

ids = []
metadatas = []
documents = []

with open("outputs/rag_chunks.jsonl", "r", encoding = "utf-8") as chunks_file :
    file_lines = chunks_file.readlines()

    for line in file_lines :

        temp = json.loads(line)
        #print(temp)
        #break
        ids.append(temp['id'])
        documents.append(temp['metin'])

        clean_meta = {k: v for k, v in temp['meta'].items() if v is not None}
        clean_meta['tur'] = temp['tur']
        metadatas.append(clean_meta)

chroma_collection.add(ids = ids, metadatas = metadatas, documents = documents) 
print(chroma_collection.count())
