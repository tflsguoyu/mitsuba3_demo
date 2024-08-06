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
        if target_path[-3:] == 'png':
            targets.append( (np.array(cv2.imread(target_path)[:,:,[2,1,0]]).astype("float32") / 255))
        elif target_path[-3:] == 'exr':
            targets.append( exrread(target_path) )
    
    diffuse_paths = glob.glob(img_dir + "diffuse_*.exr")
    envmap1_paths = glob.glob(img_dir + "envmap1_*.exr")
    envmap2_paths = glob.glob(img_dir + "envmap2_*.exr")
    
    statue0_paths = glob.glob(img_dir + "rerender1_0_*.exr")
    statue1_paths = glob.glob(img_dir + "rerender2_0_*.exr")

    frames = []
    for i, envmap1_path in enumerate(envmap1_paths):
        print(envmap1_path)
        diffuse = exrread(diffuse_paths[i])
        envmap1 = exrread(envmap1_paths[i])
        envmap2 = exrread(envmap2_paths[i])
        statue0 = exrread(statue0_paths[i])
        statue1 = exrread(statue1_paths[i])
        
        plt.figure(figsize=(24, 12))
        plt.subplot(241)
        plt.imshow(targets[0])
        plt.title('Target')
        plt.axis("off")
        
        plt.subplot(242)
        plt.imshow(statue0)
        plt.title(f'Render ({i})')
        plt.axis("off")

        plt.subplot(245)
        plt.imshow(targets[1])
        plt.title('Target')
        plt.axis("off")
        
        plt.subplot(246)
        plt.imshow(statue1)
        plt.title(f'Render ({i})')
        plt.axis("off")
        
        plt.subplot(243)
        plt.imshow(envmap1)
        plt.title(f'Envmap ({i})')
        plt.axis("off")

        plt.subplot(247)
        plt.imshow(envmap2)
        plt.title(f'Envmap ({i})')
        plt.axis("off")

        plt.subplot(144)
        plt.imshow(diffuse)
        plt.title(f'Diffuse ({i})')
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
    img2gif(["shoe/images_1_2/frame000000_cam399.png"
            ,"shoe/images_2_2/frame000000_cam399.png"]
            , "shoe_out/image12/")