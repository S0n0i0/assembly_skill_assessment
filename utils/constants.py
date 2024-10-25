from utils.classes import LogManager

log_manager = LogManager()

trainval_first_line = [
    "id",
    "video",
    "start_frame",
    "end_frame",
    "action_id",
    "verb_id",
    "noun_id",
    "action_cls",
    "verb_cls",
    "noun_cls",
    "toy_id",
    "toy_name",
    "is_shared",
    "is_rgb",
]
first_lines = {
    "splits": {
        "train": trainval_first_line,
        "validation": trainval_first_line,
        "trainval": trainval_first_line,
        "validation_challenge": trainval_first_line,
        "test": [
            "id",
            "video",
            "start_frame",
            "end_frame",
            "is_shared",
            "is_rgb",
        ],
        "test_challenge": [
            "id",
            "video",
            "start_frame",
            "end_frame",
            "is_shared",
            "is_rgb",
        ],
    },
    "actions": [
        "id",
        "action_id",
        "verb_id",
        "noun_id",
        "action_cls",
        "verb_cls",
        "noun_cls",
    ],
    "offsets": ["id", "video", "start_frame"],
    "actions_mapping": ["id", "old_action_id", "new_action_id"],
}
