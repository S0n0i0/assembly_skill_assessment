from utils.classes import LogManager

log_manager = LogManager()

fine_trainval_first_line = [
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
skill_trainval_first_line = ["id", "video", "start_frame", "end_frame", "skill_level"]
first_lines = {
    "splits": {
        "fine": {
            "train": fine_trainval_first_line,
            "validation": fine_trainval_first_line,
            "trainval": fine_trainval_first_line,
            "validation_challenge": fine_trainval_first_line,
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
        "skill": {
            "train": skill_trainval_first_line,
            "validation": skill_trainval_first_line,
            "trainval": skill_trainval_first_line,
            "test": ["id", "video", "start_frame", "end_frame"],
        },
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
