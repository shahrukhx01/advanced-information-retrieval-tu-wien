from dataloader import AmazonProductDataloader
from inverted_index import InvertedIndex
from tf_idf_search_relevance import TFIDFSearchRelevance
from bm25_search_relevance import BM25SearchRelevance

if __name__ == "__main__":
    dataloader = AmazonProductDataloader(
        file_path="dataset/amazon_products_short.csv",
        text_field="description",
        id_field="id"
    )
    inverted_index = InvertedIndex(dataloader)
    
    #relevance_scorer = TFIDFSearchRelevance(inverted_index)
    relevance_scorer = BM25SearchRelevance(inverted_index)
    query = input("input query: ")
    documents = relevance_scorer.score_query(query=query, k=20)
    print(query)
    print("------------------")
    if documents:
        print(dataloader.dataset[dataloader.dataset.id.isin(documents)].description.values)