from .helper import init_logger
from vphoberttagger.models import *
from .processor import convert_word_segment_examples_features, convert_syllable_examples_features, convert_word_segment_examples_features_from_jsonl
from datetime import datetime


LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

LABEL2ID_VLSP2016 = ['O', 'B-LOCATION-GPE', 'I-LOCATION-GPE', 'B-QUANTITY-NUM', 'B-EVENT-CUL', 'I-EVENT-CUL', 'B-DATETIME', 'I-DATETIME', 'B-DATETIME-DATERANGE', 'I-DATETIME-DATERANGE', 'B-PERSONTYPE', 'B-PERSON', 'B-QUANTITY-PER', 'I-QUANTITY-PER', 'B-ORGANIZATION', 'B-LOCATION-GEO', 'I-LOCATION-GEO', 'B-LOCATION-STRUC', 'I-LOCATION-STRUC', 'B-PRODUCT-COM', 'I-PRODUCT-COM', 'I-ORGANIZATION', 'B-DATETIME-DATE', 'I-DATETIME-DATE', 'B-QUANTITY-DIM', 'I-QUANTITY-DIM', 'B-PRODUCT', 'I-PRODUCT', 'B-QUANTITY', 'I-QUANTITY', 'B-DATETIME-DURATION', 'I-DATETIME-DURATION', 'I-PERSON', 'B-QUANTITY-CUR', 'I-QUANTITY-CUR', 'B-DATETIME-TIME', 'B-QUANTITY-TEM', 'I-QUANTITY-TEM', 'B-DATETIME-TIMERANGE', 'I-DATETIME-TIMERANGE', 'B-EVENT-GAMESHOW', 'I-EVENT-GAMESHOW', 'B-QUANTITY-AGE', 'I-QUANTITY-AGE', 'B-QUANTITY-ORD', 'I-QUANTITY-ORD', 'B-PRODUCT-LEGAL', 'I-PRODUCT-LEGAL', 'I-PERSONTYPE', 'I-DATETIME-TIME', 'B-LOCATION', 'B-ORGANIZATION-MED', 'I-ORGANIZATION-MED', 'B-URL', 'B-PHONENUMBER', 'B-ORGANIZATION-SPORTS', 'I-ORGANIZATION-SPORTS', 'B-EVENT-SPORT', 'I-EVENT-SPORT', 'B-SKILL', 'I-SKILL', 'B-EVENT-NATURAL', 'I-LOCATION', 'I-EVENT-NATURAL', 'I-QUANTITY-NUM', 'B-EVENT', 'I-EVENT', 'B-ADDRESS', 'I-ADDRESS', 'B-IP', 'I-IP', 'I-PHONENUMBER', 'B-EMAIL', 'I-EMAIL', 'I-URL', 'B-ORGANIZATION-STOCK', 'B-DATETIME-SET', 'I-DATETIME-SET', 'B-PRODUCT-AWARD', 'I-PRODUCT-AWARD', 'B-MISCELLANEOUS', 'I-MISCELLANEOUS', 'I-ORGANIZATION-STOCK', 'B-LOCATION-GPE-GEO']
LABEL2ID_VLSP2018 = ['O', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-LOCATION', 'I-LOCATION', 'B-PERSON', 'I-PERSON',
                     'B-MISCELLANEOUS', 'I-MISCELLANEOUS']
LABEL2ID_COVID19 = ['O', 'B-AGE', 'B-DATE', 'B-GENDER', 'B-JOB', 'B-LOCATION', 'B-NAME', 'B-ORGANIZATION', 'B-PATIENT_ID',
                    'B-SYMPTOM_AND_DISEASE', 'B-TRANSPORTATION', 'I-AGE', 'I-DATE', 'I-JOB', 'I-LOCATION', 'I-NAME',
                    'I-ORGANIZATION', 'I-PATIENT_ID', 'I-SYMPTOM_AND_DISEASE', 'I-TRANSPORTATION']

LABEL2ID_BDS = ["O", "B-transaction", "I-transaction", "B-real_estate_type", "I-real_estate_type",
                "B-real_estate_sub_type", "I-real_estate_sub_type", "B-price", "I-price", "B-area", "I-area",
                "B-direction", "I-direction", "B-street", "I-street", "B-ward", "I-ward", "B-district", "I-district",
                "B-city", "I-city", "B-email", "I-email", "B-phone", "I-phone", "B-usage", "I-usage", "B-floor",
                "I-floor", "B-bath_room", "I-bath_room", "B-living_room", "I-living_room", "B-bed_room", "I-bed_room",
                "B-position", "I-position", "B-author", "I-author", "B-project_owner", "I-project_owner",
                "B-project_name", "I-project_name", "B-front_length", "I-front_length", "B-road_width", "I-road_width",
                "B-surrounding", "I-surrounding", "B-legal", "I-legal", "B-house_number", "I-house_number"]

PROCESSOR_MAPPING = {
    'vinai/phobert-base': convert_word_segment_examples_features,
    'FPTAI/vibert-base-cased': convert_syllable_examples_features,
    'bert-base-multilingual-cased': convert_syllable_examples_features,
    'vinai/phobert-base/jsonl': convert_word_segment_examples_features_from_jsonl,
}

MODEL_MAPPING = {
    'vinai/phobert-base': {
        'softmax': PhoBertSoftmax,
        'crf': PhoBertCrf,
        'lstm_crf': PhoBertLstmCrf
    },
    'FPTAI/vibert-base-cased': {
        'softmax': viBertSoftmax,
        'crf': viBertCrf,
        'lstm_crf': viBertLstmCrf
    },
    'bert-base-multilingual-cased': {
        'softmax': BertSoftmax,
        'crf': BertCrf,
        'lstm_crf': BertLstmCrf
    }
}

LABEL_MAPPING = {
    'vlsp2016': {
        'label2id': LABEL2ID_VLSP2016,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2016)},
        'header': ['token', 'pos', 'chunk', 'ner']
    },
    'vlsp2018_l1': {
        'label2id': LABEL2ID_VLSP2018,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2018)},
        'header': ['token', 'ner', 'tmp1', 'tmp2']
    },
    'vlsp2018_l2': {
        'label2id': LABEL2ID_VLSP2018,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2018)},
        'header': ['token', 'tmp1', 'ner', 'tmp2']
    },
    'covid19': {
        'label2id': LABEL2ID_VLSP2018,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2018)},
        'header': ['token', 'tmp1', 'ner', 'tmp2']
    },
    'bds2022': {
        'label2id': LABEL2ID_BDS,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_BDS)},
        'header': 'jsonl'
    }
}
