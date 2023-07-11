from uuid import getnode as get_mac
mac = get_mac()
print("MAC is " , mac)
phq_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/1_P_ET_questionnaire.csv"
if mac in [218502686202210, 51664183507427, 62438188057035]:
    analysis_artifacts_dir = "/Users/orenkobo/Desktop/PhD/Aim1/Analysis_artifacts/"
    artifacts_dir = "/Users/orenkobo/Desktop/PhD/Aim1/Artifacts/"
    phq_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/1_P_ET_questionnaire.csv"
    perm_benchmark_df_fn = "/Users/orenkobo/Desktop/PhD/Aim1/Analysis_artifacts/100FOLDS_7-8_BENCHMARK_1000.csv"
    df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL.csv"

    # perm_benchmark_df_fn = "/Users/orenkobo/Desktop/PhD/Aim1/Analysis_artifacts/1636924869_all_res_df_iterall_PERM.csv"
    # df_fn = artifacts_dir + "df_new_full_unsegmented_alldata2.csv"
    # df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL_nopriorscalr.csv"


if mac == 30471595735743:
    analysis_artifacts_dir = "/export/home/orenkobo/Aim1/paper_analysis/Analysis_artifacts"
    artifacts_dir = "/export/home/orenkobo/Aim1/paper_analysis/Artifacts/"
    # df_fn = artifacts_dir + "df_new_full_unsegmented_alldata2.csv"
    phq_fn = "/export/home/orenkobo/Aim1/paper_analysis/1_P_ET_questionnaire.csv"
    perm_benchmark_df_fn = "/export/home/orenkobo/Aim1/paper_analysis/Analysis_artifacts/PERM_benchmark_100_df.csv"
    df_fn = "/export/home/orenkobo/Aim1/paper_analysis/Artifacts/df_new_full__unsegmented_alldata_new_FINAL.csv"
