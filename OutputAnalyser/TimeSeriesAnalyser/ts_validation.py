from OutputAnalyser.TimeSeriesAnalyser import bottom_len_cutoff, upper_len_cutoff

def samples_validation(df, upper_thr = upper_len_cutoff, lower_thr = bottom_len_cutoff):
    return df[(df.x_gaze_len < upper_thr) & (df.x_gaze_len > lower_thr)]