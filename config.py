import os
import os.path as osp
from os.path import dirname, abspath

MIN_PIXELS = 25**2   # Ignore this mask if it contains under these many pixels

SIZE_TO_ATTR_ID = {
    'small': [u'a106_address_current_all',
                 u'a107_address_home_all',
                 u'a108_license_plate_all',
                 u'a111_name_all',
                 u'a24_birth_date',
                 u'a49_phone',
                 u'a73_landmark',
                 u'a82_date_time',
                 u'a8_signature',
                 u'a90_email',],
    'medium': [u'a105_face_all',
                 u'a109_person_body',
                 u'a110_nudity_all',
                 u'a26_handwriting',
                 u'a30_credit_card',
                 u'a39_disability_physical',
                 u'a43_medicine',
                 u'a7_fingerprint', ],
    'large': [u'a31_passport',
                 u'a32_drivers_license',
                 u'a33_student_id',
                 u'a35_mail',
                 u'a37_receipt',
                 u'a38_ticket',
              ],
}

MODE_TO_ATTR_ID = {
    'textual': [u'a106_address_current_all',
                 u'a107_address_home_all',
                 u'a111_name_all',
                 u'a24_birth_date',
                 u'a49_phone',
                 u'a73_landmark',
                 u'a82_date_time',
                 u'a90_email',],
    'visual': [u'a105_face_all',
                 u'a108_license_plate_all',
                 u'a109_person_body',
                 u'a110_nudity_all',
                 u'a26_handwriting',
                 u'a39_disability_physical',
                 u'a43_medicine',
                 u'a7_fingerprint',
                 u'a8_signature',],
    'multimodal': [u'a30_credit_card',
                 u'a31_passport',
                 u'a32_drivers_license',
                 u'a33_student_id',
                 u'a35_mail',
                 u'a37_receipt',
                 u'a38_ticket',],
}

ATTR_IDS = MODE_TO_ATTR_ID['textual'] + MODE_TO_ATTR_ID['visual'] + MODE_TO_ATTR_ID['multimodal']
TEXT_ATTR = MODE_TO_ATTR_ID['textual']

SAFE_ATTR = 'a0_safe'

IGNORE_ATTR = [
    'a70_education_history',
    'a29_ausweis',
    'a18_ethnic_clothing',
    'a85_username',
]

class Paths:
    CONFIG_PATH = abspath(__file__)
    SRC_ROOT = dirname(CONFIG_PATH)
    PROJECT_ROOT = dirname(SRC_ROOT)

    IMAGES_PATH = osp.join(PROJECT_ROOT, 'images')

    ANNO_ROOT = osp.join(PROJECT_ROOT, 'annotations', '2017_v1')
    ANNO_EXTRA_ROOT = osp.join(PROJECT_ROOT, 'annotations-extra')
    TRAIN_ANNO_PATH = osp.join(ANNO_ROOT, 'train2017.json')
    TRAINVAL_ANNO_PATH = osp.join(ANNO_ROOT, 'trainval2017.json')
    VAL_ANNO_PATH = osp.join(ANNO_ROOT, 'val2017.json')
    TEST_ANNO_PATH = osp.join(ANNO_ROOT, 'test2017.json')

    CACHE_PATH = osp.join(SRC_ROOT, 'cache')