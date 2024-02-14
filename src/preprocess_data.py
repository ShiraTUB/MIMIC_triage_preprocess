# Before running: 1) pip install scispacy 2) pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
# ast.literal_eval performs eval that converts strings back to tuples in our csv
import ast
import re

import pandas as pd
import random

from nlp_utils import init_scispacy, find_similarity


def umls_symptom_mapper(similarity_data_path: str = None):
    triage_csv_path = '../data/triage-raw.csv'
    triage_df = pd.read_csv(triage_csv_path)
    triage_df = triage_df.drop(['Icd10'], axis=1)

    # Replace abbreviation symptom notes
    n_v_d_pattern = re.compile(r'\bn/v/d\b', flags=re.IGNORECASE)
    n_v_pattern = re.compile(r'\bn/v\b', flags=re.IGNORECASE)
    abd_pattern = re.compile(r'\babd\b', flags=re.IGNORECASE)
    c_p_pattern = re.compile(r'\bcp\b', flags=re.IGNORECASE)

    triage_df['chiefcomplaint'] = triage_df['chiefcomplaint'].str. \
        replace(n_v_d_pattern, 'nausea and vomiting and dehydration', regex=True). \
        replace(n_v_pattern, 'nausea and vomiting', regex=True). \
        replace(abd_pattern, 'abdominal', regex=True). \
        replace(c_p_pattern, 'chest pain', regex=True)

    unique_triage_df = triage_df.drop_duplicates(subset=['chiefcomplaint'])

    # Either load pre-computed UMLS codes or extract named entities again using Scispacy
    if similarity_data_path:
        similarity_df = pd.read_csv(similarity_data_path, converters={"chiefcomplaint": ast.literal_eval})
        similarity_series = similarity_df['chiefcomplaint']
    else:
        init_scispacy()
        similarity_series = unique_triage_df['chiefcomplaint'].apply(find_similarity)

    # Separate similarity_series into UMLS codes and UMLS canonical names series
    codes_list_series = similarity_series.apply(
        lambda x: [umls_triplet[0] for umls_triplet in x if len(umls_triplet) == 3])
    canonical_names_list_series = similarity_series.apply(
        lambda x: [umls_triplet[1] for umls_triplet in x if len(umls_triplet) == 3])
    scores_list_series = similarity_series.apply(
        lambda x: [umls_triplet[2] for umls_triplet in x if len(umls_triplet) == 3])

    # Merge found codes and names back to complete Dataframe
    triage_df['umls_code_list'] = triage_df['chiefcomplaint'].map(
        dict(zip(unique_triage_df.chiefcomplaint, codes_list_series)))
    triage_df['umls_canonical_name_list'] = triage_df['chiefcomplaint'].map(
        dict(zip(unique_triage_df.chiefcomplaint, canonical_names_list_series)))
    triage_df['umls_scores_list'] = triage_df['chiefcomplaint'].map(
        dict(zip(unique_triage_df.chiefcomplaint, scores_list_series)))

    # Remove rows without UMLS symptoms found
    triage_df = triage_df[triage_df.astype(str)['umls_code_list'] != '[]']

    # Merge the diagnosis_df with triage_df
    diagnosis_df = pd.read_csv('../data/diagnosis.csv')
    merged_df = pd.merge(triage_df, diagnosis_df, on=['subject_id', 'stay_id'])

    return merged_df


def convert_icd(df: pd.DataFrame):
    icd_9_to_10_df = pd.read_csv('../data/icd9to10_dict.csv')
    icd_dict = dict(zip(icd_9_to_10_df['icd9'], icd_9_to_10_df['icd10']))
    new_icd_column = df['icd_code'].map(icd_dict).fillna(df['icd_code'])
    df['icd_code'] = new_icd_column
    df.to_csv('temp')
    merged_df = df.drop(columns=['icd_version']).dropna()

    return merged_df


def preprocess_df(df: pd.DataFrame):
    # Keep only the first and second seq number rows
    df = df[df['seq_num'] <= 2]
    df = df.iloc[:, 1:]
    df = df.iloc[:, 1:]

    # Remove any rows where the ICD diagnosis code appears less than 15 times in the overall dataset
    icd_code_value_counts = df['icd_code'].value_counts()
    icd_code_value_counts_over_min = icd_code_value_counts[icd_code_value_counts > 100]

    df = df[df['icd_code'].isin(icd_code_value_counts_over_min.index)]
    df = df.reset_index().iloc[:, 1:]
    df = df.reset_index()

    # Remove any rows where all the UMLS code list appear less than 15 times in the overall dataset
    umls_codes_list = [element for sublist in list(df['umls_code_list']) for element in sublist]
    umls_code_list_value_counts = pd.Series(umls_codes_list).value_counts()
    umls_code_list_value_counts_under_min = umls_code_list_value_counts[umls_code_list_value_counts < 100]
    umls_idx_to_remove = umls_code_list_value_counts_under_min.index.tolist()

    for index, row in df.iterrows():
        if any(item in umls_idx_to_remove for item in row['umls_code_list']):
            df = df.drop(index)

    df = df.reset_index().iloc[:, 1:]
    df = df.reset_index()

    # thin out too frequent diagnosis
    icd_idx_list = list(df[df['icd_code'] == 'R079'].index)
    df = df.drop(df.index[icd_idx_list[:: 2]])

    df = df.reset_index().iloc[:, 1:]
    df = df.reset_index()

    # thin out too frequent symptoms
    icd_code_value_counts_over_max = umls_code_list_value_counts[umls_code_list_value_counts > 12000]
    umls_to_remove = icd_code_value_counts_over_max.index.tolist()
    idx_to_remove = []

    for index, row in df.iterrows():
        for i, umls in enumerate(umls_to_remove):
            if umls in row['umls_code_list']:
                idx_to_remove[i].append(index)

    # balance relevant entries
    l1 = random.choices(idx_to_remove[0], k=18000)
    l2 = random.choices(idx_to_remove[1], k=18000)
    l3 = random.choices(idx_to_remove[2], k=15000)
    l4 = random.choices(idx_to_remove[3], k=15000)
    l5 = random.choices(idx_to_remove[4], k=14000)
    l6 = random.choices(idx_to_remove[5], k=14000)
    l7 = random.choices(idx_to_remove[6], k=12000)
    l8 = random.choices(idx_to_remove[7], k=12000)

    idxs_to_remove = set(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)

    df = df.drop(list(idxs_to_remove))
    df = df.reset_index().iloc[:, 1:]
    df = df.reset_index()

    return df


def get_diagnosis_descriptions():
    outputs_df = pd.read_csv('../data/empty_output.csv').rename(
        columns={'Unnamed: 0': 'diagnosis', '13439': 'zero_colum'})
    outputs_df = outputs_df.drop(columns=['zero_colum'])
    outputs_df['diagnosis'] = outputs_df['diagnosis'].map(lambda x: x.lstrip('icd_code_'))

    helper_df = pd.read_csv('../data/icd9to10_dict.csv')

    title_dict = dict(zip(helper_df['icd10'], helper_df['Description']))
    new_col = outputs_df['diagnosis'].map(title_dict).fillna(0)

    outputs_df['diag_title'] = new_col
    outputs_df.to_csv('../data/temp_diag_dict.csv')


if __name__ == "__main__":
    # get_diagnosis_descriptions()

    triage_mapped_df = umls_symptom_mapper()
    final_df = preprocess_df(triage_mapped_df)
    final_df.to_csv('../data/filtered_data_versionx.csv')

    patients_df = pd.read_csv('../data/patients.csv')
    df_with_patients = pd.merge(final_df, patients_df, on=['subject_id'])
    df_with_patients[
        ['chiefcomplaint', 'umls_code_list', 'umls_canonical_name_list', 'pain', 'gender', 'anchor_age', 'icd_code',
         'icd_title']].to_csv('..data/balanced_data_versionx.csv')
    print('end')
