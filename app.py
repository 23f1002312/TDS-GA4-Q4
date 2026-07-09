from fastapi import FastAPI
import pandas as pd
import numpy as np
import json

app = FastAPI()

# -----------------------
# Load data once
# -----------------------

documents = pd.read_csv("documents.csv")

with open("embeddings.json") as f:
    embeddings = json.load(f)

with open("reranker_scores.json") as f:
    reranker_scores = json.load(f)


# -----------------------
# Cosine similarity
# -----------------------

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# -----------------------
# Metadata filtering
# -----------------------

def apply_filter(df, filt):

    result = df.copy()

    for key, value in filt.items():

        # exact match
        if not isinstance(value, dict):
            result = result[result[key] == value]

        else:

            if "gte" in value:
                result = result[result[key] >= value["gte"]]

            if "lte" in value:
                result = result[result[key] <= value["lte"]]

            if "in" in value:
                result = result[result[key].isin(value["in"])]

    return result


# -----------------------
# API
# -----------------------

@app.post("/vector-search")
def vector_search(request: dict):

    query_id = request["query_id"]
    query_vector = request["query_vector"]
    top_k = request["top_k"]
    rerank_top_n = request["rerank_top_n"]
    filt = request.get("filter", {})

    # -----------------------
    # Stage 1
    # -----------------------

    filtered = apply_filter(documents, filt)

    scores = []

    for _, row in filtered.iterrows():

        doc = row["doc_id"]

        sim = cosine_similarity(
            query_vector,
            embeddings[doc]
        )

        scores.append((doc, sim))

    scores.sort(
        key=lambda x: (-x[1], x[0])
    )

    candidates = [d for d, _ in scores[:top_k]]

    # -----------------------
    # Stage 2
    # -----------------------

    lookup = reranker_scores.get(query_id, {})

    ranked = []

    for doc in candidates:

        ranked.append(
            (
                doc,
                lookup.get(doc, 0)
            )
        )

    ranked.sort(
        key=lambda x: (-x[1], x[0])
    )

    return {
        "matches":
        [d for d, _ in ranked[:rerank_top_n]]
    }
