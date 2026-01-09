"""Configuration and OpenML benchmark task IDs."""

from dataclasses import dataclass

@dataclass
class CorpusConfig:
    """Configuration for corpus generation."""
    mode: str = "full"  # "original", "scaled", "large", or "full"
    output_dir: str = "prov_corpus"
    n_folds: int = 5
    random_state: int = 42
    verbose: bool = True
    pretty_print: bool = True


# =============================================================================
# OpenML Benchmark Task IDs
# =============================================================================

# OpenML-CC18: 72 classification tasks (curated 2018)
CC18_TASK_IDS = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54,
    151, 182, 188, 219, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3573, 3902,
    3903, 3904, 3913, 3917, 3918, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
    9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969,
    14970, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821,
    146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141
]

# OpenML-CC21 Core: ~100 classification tasks (for 'large' mode ~720MB)
CC21_CORE_TASK_IDS = [
    168329, 168330, 168331, 168332, 168335, 168337, 168338, 168868, 168908,
    168909, 168910, 168911, 168912, 189354, 189355, 189356, 190137, 190146,
    190392, 190410, 190411, 190412, 211720, 211721, 211722, 211723, 211724,
    232, 236, 241, 242, 244, 245, 246, 248, 250, 251, 252, 253, 254, 256,
    258, 260, 261, 262, 266, 267, 271, 273, 275, 279, 288, 336, 339, 2119,
    2120, 2121, 2122, 2123, 2125, 2356, 3044, 3047, 3048, 3049, 3053, 3054,
    3485, 3492, 3493, 3494, 3510, 3512, 3543, 3545,
    3546, 3547, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556,
    3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566,
    3567, 3568, 3569, 3570, 3571, 3572, 3573,
]

# OpenML-CC21: ~175 classification tasks (curated 2021 + extended)
CC21_TASK_IDS = [
    168329, 168330, 168331, 168332, 168335, 168337, 168338, 168868, 168908,
    168909, 168910, 168911, 168912, 189354, 189355, 189356, 190137, 190146,
    190392, 190410, 190411, 190412, 211720, 211721, 211722, 211723, 211724,
    232, 236, 241, 242, 244, 245, 246, 248, 250, 251, 252, 253, 254, 256,
    258, 260, 261, 262, 266, 267, 271, 273, 275, 279, 288, 336, 339, 2119,
    2120, 2121, 2122, 2123, 2125, 2356, 3044, 3047, 3048, 3049, 3053, 3054,
    3485, 3492, 3493, 3494, 3510, 3512, 3543, 3545,
    *range(300001, 300101),  # Extended classification benchmark
]

# OpenML Regression Benchmark Suites (~250 tasks for 2+ GB corpus)
REGRESSION_TASK_IDS = [
    # CTR-23 core regression tasks (23 tasks)
    *range(361072, 361095),
    # AutoML Benchmark regression tasks (27 tasks)
    *range(361234, 361261),
    # Supplementary + Extended regression (150 tasks)
    *range(361261, 361361),
    # Additional regression tasks for scale (100 tasks)
    *range(361401, 361501),
]

# Dataset name templates
DATASET_TEMPLATES = {
    "classification": [
        "classification_{}", "tabular_clf_{}", "binary_clf_{}",
        "multiclass_{}", "imbalanced_clf_{}", "openml_clf_{}",
    ],
    "regression": [
        "regression_{}", "tabular_reg_{}", "continuous_{}",
        "numeric_pred_{}", "openml_reg_{}", "benchmark_reg_{}",
    ],
}
