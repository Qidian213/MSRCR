import os
import cv2
import retinex

config = {
            "sigma_list": [15, 80, 200],##多尺度高斯模糊sigma值
            "G" : 5.0,                  ##增益
            "b" : 25.0,                 ##偏差
            "alpha": 125.0,
            "beta" : 46.0,
            "low_clip"  : 0.01,
            "high_clip" : 0.99
        }

data_dir = '/data/Dataset/DarkFace/image/'
dst_dir  = '/data/Dataset/DarkFace/Image_MSRCR/'

files = os.listdir(data_dir)
for file in files:
    data_path = data_dir + file #'/data/Dataset/DarkFace/image/16.png'

    img = cv2.imread(data_path)

    print('msrcr processing......')
    img_msrcr = retinex.MSRCR(img,config['sigma_list'],config['G'],config['b'],config['alpha'],config['beta'],config['low_clip'],config['high_clip'])
    cv2.imwrite(dst_dir + file,img_msrcr);

    # print('amsrcr processing......')
    # img_amsrcr = retinex.automatedMSRCR(img,config['sigma_list'])
    # cv2.imwrite('AutomatedMSRCR_retinex.png', img_amsrcr)

    # print('msrcp processing......')
    # img_msrcp = retinex.MSRCP(img, config['sigma_list'], config['low_clip'], config['high_clip'])    
    # cv2.imwrite('MSRCP.png', img_msrcp)
