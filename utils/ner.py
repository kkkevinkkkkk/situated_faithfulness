from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from span_marker import SpanMarkerModel
from collections import defaultdict
from .utils import normalize_answer

class NERModel:
    # options for model_name:
    # - Babelscape/wikineural-multilingual-ner
    # - tomaarsen/span-marker-roberta-large-ontonotes5
    def __init__(self,
                 model_name="tomaarsen/span-marker-roberta-large-ontonotes5"):

        self.model_name = model_name
        if model_name == "Babelscape/wikineural-multilingual-ner":
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = None
        if "span-marker" in model_name:
            self.model = SpanMarkerModel.from_pretrained(model_name)
            self.model.cuda()
        else:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

        self.entity_groups = {
            "CARDINAL": "Cardinal value",
            "DATE": "Date and times",
            "EVENT": "Event name",
            "FAC": "Building name",
            "GPE": "Geo-political entity",
            "LANGUAGE": "Language name",
            "LAW": "Law name",
            "LOC": "Location name",
            "MONEY": "Monetary value",
            "NORP": "Affiliation",
            "ORDINAL": "Ordinal value",
            "ORG": "Organization name",
            "PERCENT": "Percentage",
            "PERSON": "Person name",
            "PRODUCT": "Product name",
            "QUANTITY": "Quantity value",
            "TIME": "Time value",
            "WORK_OF_ART": "Work of art name",
        }
        self.meaning_to_group = {v: k for k, v in self.entity_groups.items()}
        self.merged_entity_groups_map = {
            "Number": ["CARDINAL", "QUANTITY", "ORDINAL", "PERCENT", "MONEY"],
            "Date_and_times": ["DATE", "TIME"],
            "Location": ["GPE", "LOC", "FAC"],
            "Person": ["PERSON"],
            "Organization": ["ORG", "NORP"],
            "Event": ["EVENT"],
            "Language": ["LANGUAGE"],
            "Law": ["LAW"],
            "Product": ["PRODUCT"],
            "Work_of_art": ["WORK_OF_ART"],
        }
        self.merged_entity_groups = {
            "Number": "This category includes all entities that represent numerical information, such as cardinal numbers, ordinal numbers, percentages, and monetary values.",
            "Date_and_times": "This category includes all entities that are time-related, such as dates and times.",
            "Location": "This category includes all entities that are location-related, such as countries, cities, buildings and other geographical locations.",
            "Person": "This category includes all entities that are person-related, such as names of people.",
            "Organization": "This category includes all entities that are organization-related, such as names of organizations, affiliations.",
            "Event": "This category includes all entities that are event-related.",
            "Language": "This category includes all entities that are language-related.",
            "Law": "This category includes all entities that are law-related.",
            "Product": "This category includes all entities that are product-related.",
            "Work_of_art": "This category includes all entities that are work of art-related.",
        }


    def map_entities(self, entities):
        new_entities = []
        for entity in entities:
            entity['entity_group'] = entity['label']
            for group, subgroup in self.merged_entity_groups_map.items():
                if entity['label'] in subgroup:
                    entity['entity_group'] = group
                    break

            new_entities.append(
                {
                    'entity_group': entity['entity_group'],
                    'score': entity["score"],
                    "word": entity["span"],
                    'start': entity['char_start_index'],
                    'end': entity['char_end_index'],
                }
            )
        return new_entities

    def extract_chosen_entities(self, chosen_entities_text):
        chosen_entities_text = chosen_entities_text.lower()
        entity_groups = []
        for k, v in self.merged_entity_groups.items():
            if k.lower() in chosen_entities_text:
                entity_groups.append(k)
        return entity_groups

    def __call__(self, text, *args, **kwargs):
        if "span-marker" in self.model_name:
            entites = self.model.predict(text)
            return self.map_entities(entites)
        else:
            return self.model(text, *args, **kwargs)

    def get_entities_dict(self, text, split=0):
        entities_dict = defaultdict(set)
        if split > 0:
            o = text if split == 1 else ':'.join(text.split(":")[1:])
            entities_list = [normalize_answer(entity) for entity in o.rstrip().rstrip(".").rstrip(",").split(",")]
            for entity in entities_list:
                entities_dict["ALL"].add(entity)
            return entities_dict
        entities = self.__call__(text)
        for entity in entities:
            # entities_dict[entity["entity_group"]].add(entity["word"].lower())
            entities_dict[entity["entity_group"]].add(normalize_answer(entity["word"]))

        return entities_dict
    def get_entities_list(self, text):
        entities = self.__call__(text)
        return [entity["word"] for entity in entities]