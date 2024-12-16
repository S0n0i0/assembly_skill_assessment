from utils.classes import LogManager

debug_on = True

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

fine_test_first_line = [
    "id",
    "video",
    "start_frame",
    "end_frame",
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
            "test": fine_test_first_line,
            "test_challenge": fine_test_first_line,
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
    "offsets": ["id", "video", "start_frame", "new_end_frame"],
    "actions_mapping": ["id", "old_action_id", "new_action_id"],
    "joint_coarse_actions": ["id", "video", "start_frame", "end_frame"],
}

view_dict = {
    "fixed": {
        "view1": "C10095_rgb",
        "view2": "C10115_rgb",
        "view3": "C10118_rgb",
        "view4": "C10119_rgb",
        "view5": "C10379_rgb",
        "view6": "C10390_rgb",
        "view7": "C10395_rgb",
        "view8": "C10404_rgb",
    },
    "ego": {
        "view1": ["HMC_21176875_mono10bit", "HMC_84346135_mono10bit"],
        "view2": ["HMC_21176623_mono10bit", "HMC_84347414_mono10bit"],
        "view3": ["HMC_21110305_mono10bit", "HMC_84355350_mono10bit"],
        "view4": ["HMC_21179183_mono10bit", "HMC_84358933_mono10bit"],
    },
}
usable_views = {
    "fixed": ["view3"],
    "ego": ["view1", "view2", "view3", "view4"],
}
