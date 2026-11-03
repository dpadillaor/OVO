"""Replica instance benchmark: 48 object classes (wall, floor and ceiling are not instances).

CLASS_IDS[i] is the GT class id of CLASS_NAMES[i]; GT vertex ids are class_id * 1000 + k.
HEAD, COMMON and TAIL split CLASS_NAMES by frequency.
"""

CLASS_IDS = (
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 35, 37, 44, 47, 52,
    54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91, 92, 95, 97,
    98,
)

CLASS_NAMES = (
    'basket', 'bed', 'bench', 'bin', 'blanket', 'blinds', 'book', 'bottle', 'box', 'bowl',
    'camera', 'cabinet', 'candle', 'chair', 'clock', 'cloth', 'comforter', 'cushion', 'desk',
    'desk-organizer', 'door', 'indoor-plant', 'lamp', 'monitor', 'nightstand', 'panel',
    'picture', 'pillar', 'pillow', 'pipe', 'plant-stand', 'plate', 'pot', 'sculpture', 'shelf',
    'sofa', 'stool', 'switch', 'table', 'tablet', 'tissue-paper', 'tv-screen', 'tv-stand',
    'vase', 'vent', 'wall-plug', 'window', 'rug',
)

HEAD = (
    'chair', 'blinds', 'sofa', 'table', 'rug', 'window', 'lamp', 'door', 'pillow', 'bench',
    'tv-screen', 'cabinet', 'pillar', 'blanket', 'tv-stand', 'cushion',
)

COMMON = (
    'bin', 'vent', 'bed', 'stool', 'picture', 'indoor-plant', 'desk', 'comforter', 'nightstand',
    'shelf', 'vase', 'plant-stand', 'basket', 'plate', 'monitor', 'pipe',
)

TAIL = (
    'panel', 'desk-organizer', 'wall-plug', 'book', 'box', 'clock', 'sculpture', 'tissue-paper',
    'camera', 'tablet', 'pot', 'bottle', 'candle', 'bowl', 'cloth', 'switch',
)
