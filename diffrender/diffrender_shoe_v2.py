import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

print(mi.variants())
mi.set_variant('cuda_ad_rgb')


def find_camera(c_list, c_id):
    c_id_list = []
    for c_str in c_list:
        c_id_list.append(c_str.split()[0])

    return c_list[c_id_list.index(c_id)]

def load_camera(in_dir, N=480):
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    with open(in_dir + "cameras.txt", "r") as f:
        lines = f.readlines()
    camera_list = lines[3:]

    with open(in_dir + "images.txt", "r") as f:
        lines = f.readlines()

    camera_params = []
    for i in np.arange(0, N, 40):
        j = i * 2 + 4
        str2 = lines[j]
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, image_path = str2.split()
        # str1 = camera_list[int(camera_id) - 1]
        str1 = find_camera(camera_list, camera_id)
        camera_id_tmp, _, w, h, fl, _, _, _ = str1.split()
        assert(camera_id == camera_id_tmp)
    
        world_to_camera = np.zeros((4, 4))
        world_to_camera[:3, :3] = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        world_to_camera[:3, 3] = np.array([float(tx), float(ty), float(tz)])
        world_to_camera[3, 3] = 1

        camera_to_world = np.linalg.inv(world_to_camera)
        camera_to_world[:3, :2] = -camera_to_world[:3, :2]

        fov = 2 * np.arctan(int(w) / (2 * float(fl))) * 180 / np.pi

        camera_params.append([int(w)/2, int(h)/2, fov, camera_to_world, image_path])

    return camera_params

def mse(image, image_ref):
    return dr.mean(dr.sqr(dr.clamp(image[:, :, :3], 1e-6, 1.0) ** (1 / 2.2) - image_ref))

def update_params(params, camera_params):
    width, height, fov, camera_to_world, image_path = camera_params

    params["PerspectiveCamera.film.size"] = [width, height]
    params["PerspectiveCamera.x_fov"] = fov
    params["PerspectiveCamera.to_world"] = mi.Transform4f(camera_to_world)
    params.update()
    
    return image_path

def main(in_dir, out_dir):
    camera_params_list_1 = load_camera(in_dir + "sparse_1/0/")
    N1 = len(camera_params_list_1)
    scene_list_1 = []
    params_list_1 = []
    image_ref_list_1 = []
    for i in range(N1):
        # load scene 
        scene_list_1.append(mi.load_file("shoe.xml"))
        params_list_1.append(mi.traverse(scene_list_1[i]))
        
        # update scene for current camera
        image_path = update_params(params_list_1[i], camera_params_list_1[i])
        
        # render scene under current camera to get mask
        image = mi.render(scene_list_1[i], spp=4)
        mask = np.array(image)[:, :, 3]

        # load reference image and apply mask to it
        image_ref = np.array(mi.Bitmap(in_dir + "images_1_2/" + image_path))
        image_ref = image_ref[:, :, :3].clip(0, 255).astype("float32") / 255
        image_ref[mask == 0] = 0 
        image_ref_list_1.append(mi.TensorXf(image_ref))

    camera_params_list_2 = load_camera(in_dir + "sparse_2/0/")
    N2 = len(camera_params_list_2)
    scene_list_2 = []
    params_list_2 = []
    image_ref_list_2 = []
    for i in range(N2):
        # load scene 
        scene_list_2.append(mi.load_file("shoe2.xml"))
        params_list_2.append(mi.traverse(scene_list_2[i]))

        # update scene for current camera
        image_path = update_params(params_list_2[i], camera_params_list_2[i])
        
        # render scene under current camera to get mask
        image = mi.render(scene_list_2[i], spp=4)
        mask = np.array(image)[:, :, 3]

        # load reference image and apply mask to it
        image_ref = np.array(mi.Bitmap(in_dir + "images_2_2/" + image_path))
        image_ref = image_ref[:, :, :3].clip(0, 255).astype("float32") / 255
        image_ref[mask == 0] = 0 
        image_ref_list_2.append(mi.TensorXf(image_ref))

    # exit()
    # Parameters need to be optimized and initialization
    key_envmap1 = "env1.data"
    key_envmap2 = "env2.data"
    key_diffuse = "OBJMesh.bsdf.diffuse_reflectance.data"
    key_rough = "OBJMesh.bsdf.alpha"

    ######## Optimization ##########
    opt = mi.ad.Adam(lr=0.005)
    # opt[key_envmap] = mi.TensorXf(0.5 * np.ones((200, 400, 3)))
    # opt[key_diffuse] = mi.TensorXf(0.5 * np.ones((512, 512, 3)))
    opt[key_envmap1] = mi.TensorXf(0.5 * np.random.rand(200, 400, 3))
    opt[key_envmap2] = mi.TensorXf(0.5 * np.random.rand(200, 400, 3))
    opt[key_diffuse] = mi.TensorXf(0.5 * np.random.rand(512, 512, 3))
    # opt[key_diffuse] = params_list[0][key_diffuse]
    # opt[key_rough] = 0.2
    for params in params_list_1:
        params.update(opt)
    for params in params_list_2:
        params.update(opt)

    iteration_count = 500
    errors = []
    pbar = tqdm.trange(iteration_count)
    for it in pbar:

        loss = 0

        image_list_1 = []
        for i in range(N1):
            # Perform a (noisy) differentiable rendering of the scene
            image_list_1.append(mi.render(scene_list_1[i], params_list_1[i], spp=4))
            # Evaluate the objective function from the current rendered image
            loss += mse(image_list_1[i], image_ref_list_1[i]) * (1 / N1)

        image_list_2 = []
        for i in range(N2):
            # Perform a (noisy) differentiable rendering of the scene
            image_list_2.append(mi.render(scene_list_2[i], params_list_2[i], spp=4))
            # Evaluate the objective function from the current rendered image
            loss += mse(image_list_2[i], image_ref_list_2[i]) * (1 / N2)

        if it % 10 == 0 or it == (iteration_count-1):
            for i in range(N1):
                if i == 0 or i == int(N1/2):
                    mi.util.write_bitmap(out_dir + f"rerender1_{i}_{it:03d}.exr", image_list_1[i])
            for i in range(N2):
                if i == 0 or i == int(N2/2):
                    mi.util.write_bitmap(out_dir + f"rerender2_{i}_{it:03d}.exr", image_list_2[i])

            mi.util.write_bitmap(out_dir + f"envmap1_{it:03d}.exr", opt[key_envmap1])
            mi.util.write_bitmap(out_dir + f"envmap2_{it:03d}.exr", opt[key_envmap2])
            mi.util.write_bitmap(out_dir + f"diffuse_{it:03d}.exr", opt[key_diffuse])
            plt.plot(errors)
            plt.xlabel('Iteration'); plt.ylabel('MSE(image)'); plt.title(f'Image loss');
            plt.savefig(out_dir + f"img_loss_{it:03d}.png")

        # Backpropagate through the rendering process
        dr.backward(loss)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key_envmap1] = dr.clamp(opt[key_envmap1], 0.0, 10.0)
        opt[key_envmap2] = dr.clamp(opt[key_envmap2], 0.0, 10.0)
        opt[key_diffuse] = dr.clamp(opt[key_diffuse], 0.0, 1.0)
        # opt[key_rough] = dr.clamp(opt[key_rough], 0.001, 0.7)

        # Update the scene state to the new optimized values
        for params in params_list_1:
            params.update(opt)
        for params in params_list_2:
            params.update(opt)

        # Track the difference between the current color and the true value
        pbar.set_postfix({"Loss": loss[0]})
        errors.append(loss[0])

    print('\nOptimization complete.')

if __name__ == "__main__":
    main("shoe/", "shoe_out/")  




