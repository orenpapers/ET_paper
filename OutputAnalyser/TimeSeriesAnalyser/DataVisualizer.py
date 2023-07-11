import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from OutputAnalyser.TimeSeriesAnalyser import analysis_output_dir

def visuzlise_all_samples_per_label(X_train, y_train):
    #https://tslearn.readthedocs.io/en/stable/auto_examples/classification/plot_early_classification.html#sphx-glr-auto-examples-classification-plot-early-classification-py
    n_classes = len(set(y_train))

    plt.figure(figsize = (24,18))
    colors = ['blue','orange','red','green','purple']
    y_labels = ['Minimal', 'Mild','Moderate', 'Moderately Severe', 'Severe']
    size = X_train.shape[1]

    for i, cl in enumerate(set(y_train)):
        plt.subplot(n_classes, 1, i + 1)
        for ts in X_train[y_train == cl]:
            plt.plot(ts.ravel(), color=colors[i], alpha=.3)
            plt.title("All samples - {}".format(y_labels[i]))
        plt.xlim(0, size - 1)
    plt.suptitle("Training time series")
    plt.xlabel("timestamp")
    plt.ylabel("x gaze cord")
    output_path = analysis_output_dir + "/all_samples_per_label.png"
    print("Saved per_class_timeseries_example to ", output_path)
    plt.savefig(output_path)

def per_class_timeseries_example(df, y_col = "phq_group"):
    y_vec = df[y_col]
    #https://github.com/alan-turing-institute/sktime/blob/master/examples/02_classification_univariate.ipynb
    labels, counts = np.unique(y_vec, return_counts=True)
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for label in labels:
        sen_idx = 0
        pd.Series(eval(df.x_gaze_location.loc[y_vec == label].iloc[sen_idx])).plot(ax=ax, label=f"class {label}")

    plt.fill_between(x=(0,3000),y1=6,y2=120, alpha=0.2, color='yellow')
    plt.legend()
    ax.set(title="Example time series (1 per class)", xlabel="Time")

    output_path = analysis_output_dir + "/per_class_timeseries_example.png"
    print("Saved per_class_timeseries_example to ", output_path)
    plt.savefig(output_path)