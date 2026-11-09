"""Retrieve passages and use them in a hosted answer.

Data: the separately shared class ZIP, with data/ in the working directory.
Optional dependencies for this example: pandas; huggingface_hub; sentence-transformers.
"""

# %% Read the class passages
from pathlib import Path
import pandas as pd

data_dir = Path("data")
passages = pd.read_csv(data_dir / "peloton-burton-excerpts.csv")
passage = passages.set_index("excerpt_id").loc["P04", "text"]
print(passage)
print(passages[["excerpt_id", "instructor", "sentence_start", "sentence_end"]])


# %% Configure the hosted client
import os
from huggingface_hub import InferenceClient

MODEL_ID = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
PROVIDER = os.environ.get("HF_PROVIDER", "deepinfra")
client = InferenceClient(
    provider=PROVIDER,
    api_key=os.environ["HF_TOKEN"],
)


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


# %% Answer from retrieved passages
question = (
    "Where is self-love linked to further effort, and which passages "
    "say we are already enough?"
)
question_vector = embedder.encode_query(question, normalize_embeddings=True)
selected = (document_vectors @ question_vector).argsort()[::-1][:5]
context = "\n".join(
    f"[{passages.iloc[i]['excerpt_id']}] {texts[i]}" for i in selected
)
rag_messages = [
    {"role": "system", "content": (
        "Answer only from the supplied passages. Cite excerpt IDs and "
        "quote supporting words. Separate evidence of self-demand from "
        "the stronger claim that one can never be good enough. Include "
        "counterexamples. State what the excerpts cannot establish."
    )},
    {"role": "user", "content": f"Question: {question}\nPassages:\n{context}"},
]
rag_reply = client.chat_completion(
    model=MODEL_ID, messages=rag_messages, max_tokens=512,
)
print(context)
print(rag_reply.choices[0].finish_reason)
print(rag_reply.choices[0].message.content)
