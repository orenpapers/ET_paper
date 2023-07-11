import glob
import pandas as pd
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import seaborn as sns
# from heatmappy import Heatmapper
from PIL import Image
import cv2
import numpy as np
import os
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
sns.set_style("whitegrid", {'axes.grid' : False})

def draw_heatmap_on_image(x , y , bg_image_fn, dest_fn):
    heatmap_df = pd.DataFrame(data=[x,y]).transpose()
    heatmap_df.columns = ['x','y']
    hmax = sns.kdeplot(heatmap_df.x, heatmap_df.y, cmap="Reds", shade=True, bw=.15)
    hmax.collections[0].set_alpha(0)
    print("Plotting heatmap to " , dest_fn)
    plt.savefig(dest_fn)
    plt.clf()
    plt.close()
    print("Heatmap plotted")

def draw_scanpath_on_image(scanpath_array, image_fn, sentence_idx, run_num, subject_scanpaths_dir, subject_id, trial_num,
                           rescale_image_size = True, imgToPlot_size = (1920, 1080),
                           putNumbers=False, putLines=True, animation=True,
                           left = 400, right  = 700 , up = 400 , down = 1500):

    """ This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.
        It is possible to visualize it as an animation by setting the additional
        argument animation=True.
       """

    scan_paths_dir_name = subject_scanpaths_dir + "/{}_scanpath_run#{}_trial#{}_sentence#{}/".format(subject_id, run_num, trial_num, sentence_idx)
    if not os.path.exists(scan_paths_dir_name):
        print("create dir ", scan_paths_dir_name)
        os.mkdir(scan_paths_dir_name)

    image = cv2.imread(image_fn, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rescale_image_size:
        stimulus = image[left : right, up : down]
    else:
        stimulus = image

    toPlot = [cv2.resize(stimulus, imgToPlot_size)]  # look, it is a list!

    for i in range(np.shape(scanpath_array)[0]):
        try:
            fixation = scanpath_array[i].astype(float)
            fixation = fixation.astype(int)
        except Exception as e:
            fixation = [int(float(x)) for x in scanpath_array[i]]

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        cv2.circle(frame,
                   (fixation[0], fixation[1]),
                   5, (255 , 0 , 0), 1)
        if putNumbers:
            cv2.putText(frame, str(i + 1),
                        (fixation[0], fixation[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), thickness=2)
        if putLines and i > 0:
            prec_fixation = scanpath_array[i - 1].astype(float)
            prec_fixation = prec_fixation.astype(int)
            cv2.line(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]), (0, 0, 255),
                     thickness=1, lineType=8, shift=0)

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation: toPlot.pop(0)


    for i in range(len(toPlot)):
        if (i % 10) == 0:
            figName = scan_paths_dir_name + "{}_run#{}_trial#{}_sentence#{}_scan_path_{}.jpg/".format(subject_id, run_num, trial_num, sentence_idx, i)
            imageio.imwrite(figName,  toPlot[i])
    print("Exported scanpaths to ", scan_paths_dir_name)

    return scan_paths_dir_name

def rescale_image(im, newsize = (224, 224), left = 400, top = 400, right = 1500, bottom = 700):
    from PIL import Image
    im2 = im.crop((left, top, right, bottom))
    im3 = im2.resize(newsize)
    return im3

def scanpath_to_vid(scan_paths_dir_name, trial_num, run_num, subject_id, sentence_idx):
    images_fn = glob.glob(scan_paths_dir_name + "/*.jpg")
    images_fn.sort(key = lambda x : int(x.replace(".jpg","").split("_")[-1]))

    img_array = []
    vid_suffix = "{}_run#{}_trial#{}_sentence#{}_scan_path_video.avi".format(subject_id, run_num, trial_num, sentence_idx)
    for im in images_fn:
        img = cv2.imread(im)
        height, width, layers = img.shape
        size = (width , height)
        img_array.append(img)

    video_fn = scan_paths_dir_name + vid_suffix
    out = cv2.VideoWriter(video_fn ,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    print("Created video to ", video_fn)
    return video_fn

def visualize(self, fixation_df, scanpath_df):

    fixation_specific_stim_df = fixation_df[fixation_df['stimType'] == self.stim.name]
    scanpath_specific_stim_df = scanpath_df[scanpath_df['stimType'] == self.stim.name]

    np.random.seed(400)
    fixation_sample = fixation_specific_stim_df.sample(n=1)

    sample = fixation_sample['sampleId'].values[0]
    print("Visualize sampleId - ", sample)

    sample_index = fixation_sample.index[0]
    scanpath_sample = scanpath_specific_stim_df.loc[sample_index]
    imgToPlot_size = (self.stim.size[0], self.stim.size[1])

    print('Log..... visualizing fixation map')
    self.map(fixation_sample.fixationMap.values[0], imgToPlot_size, self.stimpath, self.stim.name)
    print('Log... visualizing scanpath')
    self.scanpath(scanpath_sample.scanpath, imgToPlot_size, self.stimpath, self.stim.name, False)

    return
# import matplotlib.pyplot as plt
# img = plt.imread(frame_ps_fn)
# fig, ax = plt.subplots()
# ax.imshow(img)
#
# heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
# # plt.clf()
# ax.imshow(heatmap.T, extent=extent, origin='lower')
# # plt.show()
# # plt.savefig("")
# # ax.plot(x, y, '--', linewidth=5, color='firebrick')
# ax.plot(x,y)
# ax.imshow()
# # plt.savefig
# plt.savefig("/Users/orenkobo/Downloads/aa.png")
# pic = ""
# run_df = ""
# exit()

