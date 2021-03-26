import os
import cv2
import retinex
from multiprocessing import Pool

config = {
            "sigma_list": [15, 80, 200],##多尺度高斯模糊sigma值
            "G" : 5.0,                  ##增益
            "b" : 25.0,                 ##偏差
            "alpha": 125.0,
            "beta" : 46.0,
            "low_clip"  : 0.01,
            "high_clip" : 0.99
        }

def MSRCR_Fun(param):
    data_path, dst_path = param
    
    img       = cv2.imread(data_path)
    img_msrcr = retinex.MSRCR(img,config['sigma_list'],config['G'],config['b'],config['alpha'],config['beta'],config['low_clip'],config['high_clip'])
    cv2.imwrite(dst_path,img_msrcr);

    # img_amsrcr = retinex.automatedMSRCR(img,config['sigma_list'])
    # cv2.imwrite('AutomatedMSRCR_retinex.png', img_amsrcr)

    # img_msrcp = retinex.MSRCP(img, config['sigma_list'], config['low_clip'], config['high_clip'])    
    # cv2.imwrite('MSRCP.png', img_msrcp)

def MSRCR_pool(fps, dfps):
    pool = Pool(16)
    pool.map(MSRCR_Fun, zip(fps, dfps))
    pool.close()
    pool.join()

if __name__ == '__main__':

    data_dir = '/data/Dataset/DarkFace/image/'
    dst_dir  = '/data/Dataset/DarkFace/Image_MSRCR/'

    files = os.listdir(data_dir)
    
    source_paths = []
    dst_paths    = []
    for file in files:
        source_paths.append(data_dir + file)
        dst_paths.append(dst_dir + file)
        
    MSRCR_pool(source_paths, dst_paths)
    
    