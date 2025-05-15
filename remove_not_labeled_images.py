from codes.my_functions import *

def main():
    images_folder = 'D:/v3.0/more_images'
    masks_folder = 'D:/v3.0/more_masks'
    remove_images_without_masks(images_folder, masks_folder)

if __name__ == '__main__':
    main()