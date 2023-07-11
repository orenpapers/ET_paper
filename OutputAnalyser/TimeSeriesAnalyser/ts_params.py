
main_dir = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/"
input_csv_fn = main_dir + "/all_sentences_all_subjects_df_fn_withX.csv"
# https://colab.research.google.com/drive/1oYoCUABe4YO-jkcUAst2cDyRDVeYwHfm?authuser=2#scrollTo=e-Pk9C-uTwxD
heatmap_input_csv_fn = main_dir + "/output_for_DL_df_ALL_new.csv"
# https://colab.research.google.com/drive/13JRPv_AmwFGolzVuJ-QZG0MAKVEJMLWk?authuser=2#scrollTo=64fikUMHBaxb
analysis_output_dir = main_dir + "ts_analysis_output/"
artifacts_dir = main_dir + "Artifacts2/"
exp_output_dir = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/"
tensorboard_logs_path = "/Users/orenkobo/Desktop/PhD_new/repos/HebLingStudy/notebooks/TensorBoardLogs/"
###Hyper-parameters
bottom_len_cutoff = 1000
upper_len_cutoff = 6000
mean_pooling_factor = 5
segement_width = 500
gaussian_smoothing_factor = 3

phq2int_dict = {'Minimal':0, 'Mild' : 1 ,  'Moderate':2, 'Moderately Severe':2, 'Severe':2}

