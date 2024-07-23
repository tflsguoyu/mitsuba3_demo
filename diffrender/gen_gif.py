import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

def exrread(img_path):
    frame = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    frame = frame[:, :, :3].clip(0, 1) ** (1/2.2)
    frame = frame[:,:,[2,1,0]]
    return frame

def img2gif(target_paths, img_dir):
    targets = []
    for target_path in target_paths:
        if target_path[-3:] == 'bmp':
            targets.append( (np.array(cv2.imread(target_path)[:,:,[2,1,0]]).astype("float32") / 255) ** (1/2.2) )
        elif target_path[-3:] == 'exr':
            targets.append( exrread(target_path) )
    
    diffuse_paths = glob.glob(img_dir + "diffuse_*.exr")
    envmap_paths = glob.glob(img_dir + "envmap_*.exr")
    
    statue0_paths = glob.glob(img_dir + "statue_0_*.exr")
    statue1_paths = glob.glob(img_dir + "statue_2_*.exr")

    frames = []
    for i, envmap_path in enumerate(envmap_paths):
        print(envmap_path)
        diffuse = exrread(diffuse_paths[i])
        envmap = exrread(envmap_paths[i])
        statue0 = exrread(statue0_paths[i])
        statue1 = exrread(statue1_paths[i])
        
        plt.figure(figsize=(18, 12))
        plt.subplot(231)
        plt.imshow(targets[0])
        plt.title('Target')
        plt.axis("off")
        
        plt.subplot(234)
        plt.imshow(statue0)
        plt.title(f'Render ({i})')
        plt.axis("off")

        plt.subplot(232)
        plt.imshow(targets[1])
        plt.title('Target')
        plt.axis("off")
        
        plt.subplot(235)
        plt.imshow(statue1)
        plt.title(f'Render ({i})')
        plt.axis("off")
        
        plt.subplot(233)
        plt.imshow(diffuse)
        plt.title(f'Diffuse ({i})')
        plt.axis("off")

        plt.subplot(236)
        plt.imshow(envmap)
        plt.title(f'Envmap ({i})')
        plt.axis("off")

        plt.savefig(img_dir + "tmp.png", bbox_inches='tight')
        plt.close()

        frame = Image.open(img_dir + "tmp.png")
        frames.append(frame)
        frames[0].save(
            f"{img_dir}/00.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,
            loop=0,
        )

if __name__ == "__main__":
    img2gif(["statue_gs_refine_cxcy/images_4/cam000_frame000001.bmp"
            ,"statue_gs_refine_cxcy/images_4/cam000_frame000031.bmp"]
            , "statue_gs_refine_cxcy_out/r4_2048/")