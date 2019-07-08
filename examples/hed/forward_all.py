import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
#%matplotlib inline
import scipy.misc
import scipy.io
from PIL import Image
import scipy.io
import os
import sys
from os.path import join, splitext, split, isfile
import time

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
sys.path.insert(0, caffe_root + 'python')
import caffe

def correctFilePath(path):
  '''
  @brief correct file separators and add separator for directories at the end.
  '''
  path = path.replace('/', os.path.sep)
  path = path.replace('\\', os.path.sep)

  if os.path.isdir(path):
    if path[len(path) - 1] != os.path.sep:
      path = path + os.path.sep

  return path


def getImageFileNames(inputDir, supportedExtensions=['.png', '.jpg', '.jpeg']):
  '''
  @brief Get image files (png, jpg, jpeg) from an input directory.
  @param inputDir Input directory that contains images.
  @param supportedExtensions Only files with supported extensions are included in the final list.
  @return List of images file names.
  '''
  res = []

  for root, directories, files in os.walk(inputDir):
      for f in files:
          for extension in supportedExtensions:
            fn, ext = splitext(f.lower())

            if extension == ext:
              res.append(f)

  return res


def createDataList(inputDir, outputFileName='data.lst', supportedExtensions=['.png', '.jpg', '.jpeg']):
  '''
  @brief Get image files (png, jpg, jpeg) from an input directory and save it as pair to data_lst.txt.
  @param inputDir Input directory that contains images.
  @param supportedExtensions Only files with supported extensions are included in the final list.
  @return Found images as list.
  '''
  out = open(join(inputDir, outputFileName), "w")
  res = []
  for root, directories, files in os.walk(inputDir):
      for f in files:
          for extension in supportedExtensions:
            fn, ext = splitext(f.lower())

            if extension == ext:
              out.write('%s\n' % (f))
              res.append(f)

  out.close()
  return res

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(15, i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()

def process():
    data_root = '../../../data/HED-BSDS/'
    # data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_xyz/rgb/'
    data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_dimmed_xyz/rgb/'
    # data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_neon_xyz/rgb/'
    # data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_rpy/rgb/'
    # data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_dimmed_rpy/rgb/'
    # data_root = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_neon_rpy/rgb/'
    createDataList(data_root, 'test.lst')

    with open(data_root + 'test.lst') as f:
        test_lst = f.readlines()
        
    test_lst = [x.strip() for x in test_lst]
    im_lst = []

    for i in range(0, len(test_lst)):
        im = Image.open(data_root+test_lst[i])
        #im = im.resize((200, 20,,), Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        im_lst.append(in_)



    #remove the following two lines if testing with cpu
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # load net
    net = caffe.Net('deploy.prototxt', 'hed_pretrained_bsds.caffemodel', caffe.TEST)

    save_root = os.path.join(data_root, 'hed')
    if not os.path.exists(save_root):
      os.mkdir(save_root)

    start_time = time.time()
    timeRecords = open(join(save_root, 'hed_timeRecords.txt'), "w")
    timeRecords.write('# filename time[ms]\n')

    for idx in range(0, len(test_lst)):
        tm = time.time()
        in_ = im_lst[idx]
        in_ = in_.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W) set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        res = net.forward()
        # out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
        # out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
        # out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
        # out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
        # out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]

        elapsedTime = time.time() - tm
        timeRecords.write('%s %f\n'%(test_lst[i], elapsedTime * 1000))

        # scale_lst = [fuse]
        # edge = np.squeeze(res['sigmoid-fuse'][0, 0, :, :])
        # edge = np.squeeze(net.blobs['sigmoid-fuse'].data[0][0::])
        # edge /= edge.max()

        scipy.misc.imsave(save_root + '/' + test_lst[idx][:-4] + '.png', fuse)

        #plot_single_scale(scale_lst, 22)
        #scale_lst = [out1, out2, out3, out4, out5]
        #plot_single_scale(scale_lst, 10)

    timeRecords.close()

    diff_time = time.time() - start_time
    print('Detection took {:.3f}s per image'.format(diff_time/len(test_lst)))

def main():
  process()


if __name__ == "__main__":
  main()