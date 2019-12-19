import os
import cv2
import tqdm

import numpy as np
import pandas as pd

import keras
import tensorflow as tf

TEST_DIR = '/data/test'
RESULT_PATH = '/src/output.csv'
OUTPUT_PATH = '/data/output/output.csv'
CKPT_DIR = '/data/volume/model'
HIST_DIR = '/data/volume/history'

IMAGE_SIZE = 1024

from get_major_axis import get_major_axis

def main():
    print('Start Inference!')
    cls_df = pd.read_csv(os.path.join(HIST_DIR, 'history.csv'))
    output = pd.read_csv(RESULT_PATH)

    # inference
    cls_bestepoch = cls_df[cls_df['val_acc'] == cls_df['val_acc'].max()]['epoch'].values[0]+1
    cls_model = keras.models.load_model(os.path.join(CKPT_DIR, '{:04d}_{:.4f}.h5'.format(cls_bestepoch, cls_df['val_acc'].max())))

    total_result = []
    for i, slide_id in enumerate(output['id'].values):
        print('{}/{}'.format(i+1, len(output)), slide_id)
        img = cv2.imread(os.path.join(TEST_DIR, 'level4/Image/{}.png'.format(slide_id)))[...,::-1].astype('float32')
        img_shape = img.shape
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img /= 255

        cls_result = cls_model.predict_on_batch(img[np.newaxis,...])
        cls_result = np.argmax(np.squeeze(cls_result))

        if cls_result > 0:
            total_result.append([slide_id, 1, 1.])
        else:
            total_result.append([slide_id, 0, 0.])
        
        print(total_result[i])

    result = pd.DataFrame(data=total_result, columns=['id', 'metastasis', 'major_axis'])
    result.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()