class Paths(Object):
    CONFIG_PATH = abspath(__file__)
    SRC_ROOT = dirname(CONFIG_PATH)
    PROJECT_ROOT = dirname(SRC_ROOT)

    ANNO_ROOT = osp.join(PROJECT_ROOT, 'annotations', '2017_v1')
    TRAIN_ANNO_PATH = osp.join(ANNO_ROOT, 'train2017.json')
    VAL_ANNO_PATH = osp.join(ANNO_ROOT, 'val2017.json')
    TEST_ANNO_PATH = osp.join(ANNO_ROOT, 'test2017.json')