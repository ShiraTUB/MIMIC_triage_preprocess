import ast
import itertools
import spacy
import pandas as pd

# Package by Allen Institute for AI that contains custom pipes and models related to using spaCy for scientific documents.
# Before running: 1) pip install scispacy 2) pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
# https://github.com/allenai/scispacy
import scispacy

from spacy.tokens import Token, Doc

# The EntityLinker is a SpaCy component which performs linking to a knowledge base.
# The linker simply performs a string overlap - based search (char-3grams) on named entities.
from scispacy.linking import EntityLinker

# The AbbreviationDetector is a Spacy component which implements the abbreviation detection algorithm
from scispacy.abbreviation import AbbreviationDetector

nlp = None
linker = None


def init_scispacy():
    global nlp
    global linker

    # A full spaCy pipeline for biomedical data with a ~100k vocabulary.
    nlp = spacy.load("en_core_sci_sm")

    # Add the abbreviation pipe to the spacy pipeline.
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True, "linker_name": "umls", 'max_entities_per_mention': 10})

    # Each entity is linked to UMLS with a score (currently just char-3gram matching).
    linker = nlp.get_pipe("scispacy_linker")


# Return a list of triplet tuples (UMLS concept ID, UMLS Canonical Name, Match score) for each entity that is in the given list of UMLS types
def filter_entities_by_types_group(umls_entities_list: list, type_group: list) -> list:
    return [(umls_entity[0].concept_id, umls_entity[0].canonical_name, umls_entity[1]) for umls_entity in
            umls_entities_list if [i for i in umls_entity[0].types if i in type_group]]


def extract_umls_entities(kb_ents: list) -> list:
    # 1. T184 Sign or Symptom - return ALL and not only the first
    type_group_1 = ['T184']

    # 2. T037 Injury or Poisoning
    type_group_2 = ['T037']

    # 3. T004 Fungus | T005 Virus | T007 Bacterium | T033 Finding | T034 Laboratory or Test Result | T048 Mental or Behavioral Dysfunction
    type_group_3 = ['T004', 'T005', 'T007', 'T033', 'T034', 'T048']

    # 4. T047 Disease or Syndrome | T121 Pharmacologic Substance | T131 Hazardous or Poisonous Substance
    type_group_4 = ['T047', 'T121', 'T131']

    # Extract the UMLS Entities from each CUI value
    umls_entities_list = list(map(lambda ent: (linker.kb.cui_to_entity[ent[0]], ent[1]), kb_ents))

    type_1_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_1)
    type_2_matches_list = []
    type_3_matches_list = []
    type_4_matches_list = []

    if not type_1_matches_list:
        type_2_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_2)

        if not type_2_matches_list:
            type_3_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_3)

        if not type_3_matches_list:
            type_4_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_4)

    secondary_matches_list = list(itertools.chain(type_2_matches_list, type_3_matches_list, type_4_matches_list))

    best_fit_secondary_value = 0
    best_fit_secondary_tuple = None

    # Keep only one secondary match at the most (the one with the best match value)
    for match in secondary_matches_list:
        if match[2] > best_fit_secondary_value:
            best_fit_secondary_tuple = match
            best_fit_secondary_value = match[2]

    if best_fit_secondary_tuple:
        type_1_matches_list.append(best_fit_secondary_tuple)

    return type_1_matches_list


def find_similarity(complaint_str: str):
    if not isinstance(complaint_str, str):
        return [], []

    umls_entity_list = []
    complaint_str_parenthesis = '"{}"'.format(complaint_str)
    doc = nlp(complaint_str_parenthesis)

    for ent in doc.ents:
        if len(ent._.kb_ents) > 0 and len(ent._.kb_ents[0]) > 0:
            most_similar_list = extract_umls_entities(ent._.kb_ents)
            umls_entity_list = umls_entity_list + most_similar_list

    return umls_entity_list


def umls_code_to_canonical_name():
    init_scispacy()
    df = pd.read_csv('../data/filtered_data_4.csv.csv', converters={"umls_code_list": ast.literal_eval})

    umls_code_list = df['umls_code_list'].apply(lambda x: pd.Series(x)).stack().unique()

    entity_list = list(map(lambda x: linker.kb.cui_to_entity[x], umls_code_list.tolist()))
    entity_dict = {i.concept_id: {'canonical_name': i.canonical_name, 'aliases': i.aliases, 'definition': i.definition,
                                  'types': i.types} for i in entity_list}
    entity_df = pd.DataFrame(entity_dict).transpose()
    entity_df.to_csv('../data/umls_dict.csv')
