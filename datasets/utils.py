import os
from pathlib import Path


def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def get_image_paths_from_dir_2(img_dir, img_tag="png"):
    img_path_list = Path(img_dir).glob(f"*.{img_tag}")
    img_path_list = [str(img_path) for img_path in img_path_list]
    img_path_list = sorted(img_path_list, key=os.path.basename)
    return img_path_list
    
