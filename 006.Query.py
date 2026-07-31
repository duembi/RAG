import chromadb 
from chromadb.utils import embedding_functions

sentence_transformer_model = "distiluse-base-multilingual-cased-v1"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = sentence_transformer_model)

chroma_client = chromadb.PersistentClient(path = "./")

#şimdi de oluşturmuş olduğumuz rag_chunks koleksiyonuna bağlanacağız
chroma_collection = chroma_client.get_collection("rag_chunks", embedding_function = embedding_function)

sorgu = "Kayseri de tekstil sektörü olan osbler hangileridir"

ans = chroma_collection.query(query_texts = sorgu, n_results = 5, where = {"tur" : "osb_sektor"})
print(ans)
