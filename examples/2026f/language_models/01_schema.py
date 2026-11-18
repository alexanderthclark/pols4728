"""Subclassing, structured records, and validation.

Data: the separately shared class ZIP, with data/ in the working directory.
Optional dependencies for this example: pandas; pydantic>=2,<3.
"""

# %% Read the class passages
from pathlib import Path
import pandas as pd

data_dir = Path("data")
passages = pd.read_csv(data_dir / "peloton-burton-excerpts.csv")
passage = passages.set_index("excerpt_id").loc["P04", "text"]
print(passage)
print(passages[["excerpt_id", "instructor", "sentence_start", "sentence_end"]])


# %% Define the Annotation subclass
from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError

Presence = Literal["present", "absent", "unclear"]

class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    self_affirmation: Presence
    self_demand: Presence
    affirmation_evidence: str
    demand_evidence: str

    def evidence_matches(self, source_text):
        pairs = [
            (self.self_affirmation, self.affirmation_evidence),
            (self.self_demand, self.demand_evidence),
        ]
        for label, quote in pairs:
            if label == "present":
                if not quote.strip() or quote not in source_text:
                    return False
            elif quote != "":
                return False
        return True


# %% Inspect a worked annotation
import json

with (data_dir / "peloton-burton-worked-example.json").open(encoding="utf-8") as source:
    worked_example = json.load(source)
example = Annotation(**worked_example["annotation"])
print(example.self_demand)
print(example.model_dump())
print(example.evidence_matches(passage))
print(isinstance(example, BaseModel))

schema = Annotation.model_json_schema()
print(schema["properties"]["self_demand"])


# %% Validate JSON and inspect a wrong label
valid_json = example.model_dump_json()
parsed = Annotation.model_validate_json(valid_json)
print(parsed.model_dump())

invalid_json = valid_json.replace('"present"', '"positive"', 1)
try:
    Annotation.model_validate_json(invalid_json)
except ValidationError as error:
    print(error.errors()[0]["type"])

wrong = Annotation.model_validate({
    **example.model_dump(),
    "self_affirmation": "absent", "affirmation_evidence": "",
})
print(wrong.evidence_matches(passage))
