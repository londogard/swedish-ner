from typing import List, Union
import numpy as np
import html

def get_config():
    return {
  "_num_labels": 14,
  "architectures": ["BertForTokenClassification"],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "O",
    "1": "OBJ",
    "2": "TME",
    "3": "ORG/PRS",
    "4": "OBJ/ORG",
    "5": "PRS/WRK",
    "6": "WRK",
    "7": "LOC",
    "8": "ORG",
    "9": "PER",
    "10": "LOC/PRS",
    "11": "LOC/ORG",
    "12": "MSR",
    "13": "EVN"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "EVN": 13,
    "LOC": 7,
    "LOC/ORG": 11,
    "LOC/PRS": 10,
    "MSR": 12,
    "O": 0,
    "OBJ": 1,
    "OBJ/ORG": 4,
    "ORG": 8,
    "ORG/PRS": 3,
    "PER": 9,
    "PRS/WRK": 5,
    "TME": 2,
    "WRK": 6
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": True,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 50325
}

def group_sub_entities(entities: List[dict], tokenizer) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": tokenizer.convert_tokens_to_string(tokens)
        }

        return entity_group

def group_entities(entities: List[dict], tokenizer) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.
        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:
            is_last_idx = entity["index"] == last_idx
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [group_sub_entities(entity_group_disagg, tokenizer)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            if (
                entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ):
                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [group_sub_entities(entity_group_disagg, tokenizer)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [group_sub_entities(entity_group_disagg, tokenizer)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [group_sub_entities(entity_group_disagg, tokenizer)]

        return entity_groups


TAGGED_ENTITY = """
<mark class="entity" style="background: {color}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 3; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    {entity}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 3; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
"""

PARAGRAPH = """<p>{sentence}</p>"""

def render_ner_html_custom(
    text: str,
    predictions: List[dict],
    colors={
        "PER": "#F7FF53",
        "ORG": "#E8902E",
        "LOC": "#FF40A3",
        "MISC": "#4647EB",
        "O": "#ddd",
    },
    default_color: str = "#ddd"
) -> str:
    escaped_text = html.escape(text).replace("\n", "<br/>")
    for prediction in predictions:
        tag = prediction['entity_group']
        tag_text = TAGGED_ENTITY.format(entity=prediction['word'], label=tag, color=colors.get(tag, default_color))
        escaped_text = escaped_text.replace(prediction['word'], tag_text)

    line = PARAGRAPH.format(sentence="".join(escaped_text))

    return line