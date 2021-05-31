from collections import defaultdict

SEED = 42
DROPOUT_SEED = 42
ANALYSIS_TYPES = [
    defaultdict(list, {'name': 'LAYER_SIZE', 'values': [128, 256, 512, 1024]}),
    defaultdict(list, {'name': 'NUMBER_LAYERS', 'values': [2, 4, 6, 8, 10]}),
    defaultdict(list, {'name': 'INPUT_ORDER', 'values': [1, 2, 3, 4, 5]}),
    defaultdict(list, {'name': 'NUMBER_LABELS', 'values': [2, 6, 12, 23, 46]})
]
