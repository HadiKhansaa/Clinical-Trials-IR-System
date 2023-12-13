import numpy as np

def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.0

    discounts = np.log2(np.arange(n_relevances) + 2)  # We start counting from 1
    return np.sum((2**relevances - 1) / discounts)

def ndcg_at_k(relevances, k):
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max

# Provided relevance scores for each query
def computeNDCG(relevance_scores):

    # Calculate NDCG@5 for each query
    ndcg_scores = {}
    for i,relevances in enumerate(relevance_scores):
        ndcg_at_5 = ndcg_at_k(relevances, 10)
        ndcg_scores[i] = ndcg_at_5

    # Output NDCG@5 for each query
    for query, score in ndcg_scores.items():
        print(f"NDCG@10 for '{query}': {score:.3f}")
    return ndcg_scores

# computeNDCG([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#              [0 ,2 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,1]])