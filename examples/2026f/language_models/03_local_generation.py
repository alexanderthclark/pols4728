"""Local text generation.

Data: the separately shared class ZIP, with data/ in the working directory.
Optional dependencies for this example: pandas; transformers; torch.
"""

# %% Read the class passages
from pathlib import Path
import pandas as pd

data_dir = Path("data")
passages = pd.read_csv(data_dir / "peloton-burton-excerpts.csv")
passage = passages.set_index("excerpt_id").loc["P04", "text"]
print(passage)
print(passages[["excerpt_id", "instructor", "sentence_start", "sentence_end"]])


# %% Generate a local response
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    device=-1,
)
local_messages = [{
    "role": "user",
    "content": (
        "Does this passage explicitly affirm personal worth, self-love, "
        "or authority over one's own life? Answer present, absent, "
        f"or unclear. Passage: {passage}"
    ),
}]
local_output = generator(
    local_messages,
    max_new_tokens=64,
    do_sample=False,
)
print(local_output[0]["generated_text"][-1]["content"])
