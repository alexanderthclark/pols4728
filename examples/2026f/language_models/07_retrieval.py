"""Retrieve passages with normalized embeddings.

Data: the separately shared class ZIP, with data/ in the working directory.
Optional dependencies for this example: pandas; sentence-transformers.
"""

# %% Read the class passages
from pathlib import Path
import pandas as pd

data_dir = Path("data")
passages = pd.read_csv(data_dir / "peloton-burton-excerpts.csv")
passage = passages.set_index("excerpt_id").loc["P04", "text"]
print(passage)
print(passages[["excerpt_id", "instructor", "sentence_start", "sentence_end"]])


# %% Embed and retrieve passages
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = passages["text"].tolist()
document_vectors = embedder.encode_document(texts, normalize_embeddings=True)

query = "self-love and the obligation to improve yourself"
query_vector = embedder.encode_query(query, normalize_embeddings=True)
similarities = document_vectors @ query_vector
ranking = similarities.argsort()[::-1]

print(document_vectors.shape)
print(query_vector.shape)
for index in ranking[:3]:
    print(passages.iloc[index]["excerpt_id"], float(similarities[index]), texts[index])
