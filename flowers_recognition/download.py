# coding=utf-8

from os.path import isfile, isdir

from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not isfile("vgg16.npy"):
    with DLProgress(unit='B', unit_scale=True, miniters=1,
                    desc='VGG16 Parameters') as pbar:
        urlretrieve(
                'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
                'vgg16.npy',
                pbar.hook)
else:
    print("Parameter file already exists!")

dataset_folder_path = 'flower_photos'
if not isfile('flower_photos.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1,
                    desc='Flowers Dataset') as pbar:
        urlretrieve(
                'http://download.tensorflow.org/example_images/flower_photos.tgz',
                'flower_photos.tar.gz',
                pbar.hook)

if not isdir(dataset_folder_path):
    with tarfile.open('flower_photos.tar.gz') as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        tar.close()
