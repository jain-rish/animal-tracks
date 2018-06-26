

from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import cv2, os


file_directory = '/Users/rmillin/Documents/Insight/image_reorg/all_bear'
file_names = [f for f in listdir(file_directory) if isfile(join(file_directory, f))]

for ind1, file1 in enumerate(file_names):
    im1 = cv2.imread(join(file_directory,file1))
    f1 = plt.figure()
    try:
        plt.imshow(im1, cmap='gray')
        plt.show(f1)
        for ind2, file2 in enumerate(file_names):
            if (ind2>ind1):
                im2 = cv2.imread(join(file_directory,file2))
                f2 = plt.figure()
                try:
                    plt.imshow(im2, cmap='gray')
                    plt.show(f2)
                    print('...')
                    print(file1)
                    print(file2)
                    input("Press Enter to continue...")
                except:
                    print('fail2')
                plot.close(f2)
    except:
        print('fail1')
        plt.close(f1)


               
        
            
        
