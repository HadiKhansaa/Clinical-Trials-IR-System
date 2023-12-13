from math import log2  


def computeNDCG(processing_results):
    # Given results
    l = {}
    for mmm, results in enumerate(processing_results):

        # DCG
        dcg = sum([rel if rank == 0 else (rel / (log2(rank + 1))) for rank, rel in enumerate(results)])

        # IDCG
        ideal_order = sorted([rel for rel in results], reverse=True)
        idcg = sum([rel if rank == 0 else (rel / (log2(rank + 1))) for rank, rel in enumerate(ideal_order)])

        # NDCG
        ndcg = dcg / idcg


        print(f"NDCG: {ndcg}")
        l[mmm] = ndcg

    # s1 = 0.0
    # for key, q in l.items():
    #     s1+=q

    # print(f"means: {s1/5}")

# computeNDCG([[0, 1, 0, 0, 1, 0, 1, 0,0,0]])