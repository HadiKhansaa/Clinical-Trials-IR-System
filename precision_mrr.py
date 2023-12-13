def calculate_metrics(retrieved_documents):
    # Initialize variables
    reciprocal_rank = 0.0
    precision_at_10 = 0
    found_relevant = False

    # Iterate through the first 10 documents
    for i, doc in enumerate(retrieved_documents[:10]):
        if doc==1 or doc==2:
            if not found_relevant:
                # Calculate reciprocal rank
                reciprocal_rank = 1 / (i + 1)
                found_relevant = True

            # Count the number of relevant documents for Precision@10
            precision_at_10 += 1

    # Calculate Precision@10
    precision_at_10 /= 10

    return reciprocal_rank, precision_at_10

def compute_avg_pr_mrr(retrieved_docs):
    psum = 0.0
    rsum = 0.0
    for list in retrieved_docs:
        reciprocal_rank, precision_at_10 = calculate_metrics(list)
        psum += precision_at_10
        rsum += reciprocal_rank

    return rsum/10,psum/10

# Example usage
retrieved_docs = [
    [1, 0, 2, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 2, 1, 2, 2, 0, 1, 2],
    [2, 1, 0, 0, 2, 2, 2, 0, 0, 0],
    [2, 2, 2, 0, 2, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 2, 0, 0, 2],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
]

