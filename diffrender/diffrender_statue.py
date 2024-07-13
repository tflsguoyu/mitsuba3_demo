import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

print(mi.variants())
mi.set_variant('cuda_ad_rgb')


def load_camera(in_dir, N = 2):
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    with open(in_dir + "cameras.txt", "r") as f:
        lines = f.readlines()
    camera_list = lines[3:]

    with open(in_dir + "images.txt", "r") as f:
        lines = f.readlines()

    camera_params = []
    for i in range(N):
        j = i * 2 + 4
        str2 = lines[j]
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, image_path = str2.split()
        str1 = camera_list[int(camera_id) - 1]
        camera_id_tmp, _, w, h, fl, _, _, _ = str1.split()
        assert(camera_id == camera_id_tmp)
    
        world_to_camera = np.zeros((4, 4))
        world_to_camera[:3, :3] = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        world_to_camera[:3, 3] = np.array([float(tx), float(ty), float(tz)])
        world_to_camera[3, 3] = 1

        camera_to_world = np.linalg.inv(world_to_camera)
        camera_to_world[:3, :2] = -camera_to_world[:3, :2]

        fov = 2 * np.arctan(int(w) / (2 * float(fl))) * 180 / np.pi

        camera_params.append([int(w), int(h), fov, camera_to_world, image_path])

    return camera_params

def mse(image, image_ref):
    return dr.mean(dr.sqr(image[:, :, :3] ** (1 / 2.2) - image_ref))

def update_params(params, camera_params):
    width, height, fov, camera_to_world, image_path = camera_params

    params["PerspectiveCamera.film.size"] = [width, height]
    params["PerspectiveCamera.x_fov"] = fov
    params["PerspectiveCamera.to_world"] = mi.Transform4f(camera_to_world)
    params.update()
    
    return params, image_path

def main(in_dir, out_dir):
    camera_params_list = load_camera(in_dir + "sparse/0/")

    # load scene
    scene = mi.load_file("statue.xml")
    params = mi.traverse(scene)
    # print(params)  # you can check which para could be optimized
    # exit()

    # load reference images
    image_ref_list = []
    for i in range(len(camera_params_list)):
        params, image_path = update_params(params, camera_params_list[i])

        image = mi.render(scene, params, spp=4)
        # mi.util.write_bitmap(out_dir + "statue.exr", image)
        # exit()

        mask = np.array(image)[:, :, 3]

        image_ref = np.array(mi.Bitmap(in_dir + "images/" + image_path))
        image_ref = image_ref[:, :, :3].astype("float32") / 255
        image_ref[mask == 0] = 0 
        image_ref = mi.TensorXf(image_ref)
        image_ref_list.append(image_ref)

    # Parameters need to be optimized and initialization
    key_envmap = "EnvironmentMapEmitter.data"
    key_diffuse = "OBJMesh.bsdf.diffuse_reflectance.data"
    key_rough = "OBJMesh.bsdf.alpha"

    params[key_envmap] = mi.TensorXf(0.5 * np.ones((200, 400, 3)))
    params[key_diffuse] = mi.TensorXf(0.5 * np.ones(params[key_diffuse].shape))
    params[key_rough] = 0.1
    params.update()



    ######## Optimization ##########
    opt = mi.ad.Adam(lr=0.02)
    opt[key_envmap] = params[key_envmap]
    opt[key_diffuse] = params[key_diffuse]
    opt[key_rough] = params[key_rough]
    params.update(opt)

    iteration_count = 500

    errors = []
    pbar = tqdm.trange(iteration_count)
    for it in pbar:

        loss = 0
        image_list = []
        for i in range(len(camera_params_list)):
            params, _ = update_params(params, camera_params_list[i])

            # Perform a (noisy) differentiable rendering of the scene
            image_list.append(mi.render(scene, params, spp=4))

            # Evaluate the objective function from the current rendered image
            loss += mse(image_list[i], image_ref_list[i])

        if it % 50 == 0:
            mi.util.write_bitmap(out_dir + f"statue_{it:03d}.exr", image_list[0])
            mi.util.write_bitmap(out_dir + f"envmap_{it:03d}.exr", params[key_envmap])
            mi.util.write_bitmap(out_dir + f"diffuse_{it:03d}.exr", params[key_diffuse])
            plt.plot(errors)
            plt.xlabel('Iteration'); plt.ylabel('MSE(image)'); plt.title(f'Image loss (alpha = {params[key_rough][0]:.2f})');
            plt.savefig(out_dir + f"img_loss_{it:03d}.png")

        # Backpropagate through the rendering process
        dr.backward(loss)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key_envmap] = dr.clamp(opt[key_envmap], 0.0, 100.0)
        opt[key_diffuse] = dr.clamp(opt[key_diffuse], 0.0, 1.0)
        opt[key_rough] = dr.clamp(opt[key_rough], 0.001, 0.7)

        # Update the scene state to the new optimized values
        params.update(opt)

        # Track the difference between the current color and the true value
        pbar.set_postfix({"Loss": loss[0]})
        errors.append(loss[0])

    print('\nOptimization complete.')

    params, _ = update_params(params, camera_params_list[0])
    image_final = mi.render(scene, spp=128)

    mi.util.write_bitmap(out_dir + "statue_final.exr", image_final)
    mi.util.write_bitmap(out_dir + "envmap_final.exr", params[key_envmap])
    mi.util.write_bitmap(out_dir + "diffuse_final.exr", params[key_diffuse])
    plt.plot(errors)
    plt.xlabel('Iteration'); plt.ylabel('MSE(image)'); plt.title(f'Image loss (alpha = {params[key_rough][0]:.2f})');
    plt.savefig(out_dir + "img_loss_final.png")

if __name__ == "__main__":
    main("statue_gs_refine_cxcy/", "statue_gs_refine_cxcy_out/")  




