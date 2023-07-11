import numpy as np
import seaborn as sns
import pandas as pd

class ET_ReadingMeasuresFetcher:

    def __init__(self, all_subjects_df):
        self.df = all_subjects_df

    def add_et_reading_measures(self):
        measures_keys = ["first_fixation_duration", "second_fixation_duration", "total_gaze_duration",
                         "num_of_fixations", "word_has_first_pass_regression", "regression_path_duration",
                         "skipping_probability", "word_pupil_diameter_mean", "sentence_pupil_diameter_mean"]

        measures_col_names = []
        for sentence_condition in ["A", "B", "C", "D"]:
            for measure in measures_keys:
                measures_col_names.append(sentence_condition + "_target_word_" + measure)

        self.columns = ["level", "list_of_subject_ids","n", "trait", "state"] + measures_col_names

        rows = []

        for phq_level, phq_level_rows_df in self.df.groupby("phq_level"):
            row = [phq_level, str(list(phq_level_rows_df["subject_id"])), len(phq_level_rows_df)]
            # phq_level_rows = self.df["phq_level"] == phq_level
            mean_trait = phq_level_rows_df["trait_score"].mean()
            mean_state = phq_level_rows_df["state_score"].mean()
            row += [mean_trait, mean_state]
            for col_name in measures_col_names:
                measure_mean = phq_level_rows_df[col_name].mean()
                row.append(measure_mean)

            rows.append(row)

        df = pd.DataFrame(rows, columns=self.columns)
        return  df



