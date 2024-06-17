import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

print(mi.variants())
mi.set_variant('cuda_ad_rgb')


def main():
    # load scene
    scene = mi.load_file('xml/cbox.xml', res=128, integrator='prb')

    # generate ref image
    image_ref = mi.render(scene, spp=512)
    mi.util.write_bitmap("img_ref.png", image_ref)

    # get the paramerater to be optimized
    params = mi.traverse(scene)
    # print(params)  # you can check which para could be optimized
    # exit()
    key = 'red.reflectance.value'

    # Save the original value
    param_ref = mi.Color3f(params[key])

    def mse(image):
        return dr.mean(dr.sqr(image - image_ref))

    ######## Optimization ##########
    opt = mi.ad.Adam(lr=0.05)
    opt[key] = mi.Color3f(0.01, 0.2, 0.9)

    iteration_count = 50

    errors = []
    for it in range(iteration_count):

        # Update the scene state to the new optimized values
        params.update(opt)

        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(scene, params, spp=4)
        mi.util.write_bitmap(f"img_{it:03d}.png", image)

        # Evaluate the objective function from the current rendered image
        loss = mse(image)

        # Backpropagate through the rendering process
        dr.backward(loss)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        # Track the difference between the current color and the true value
        err_ref = dr.sum(dr.sqr(param_ref - params[key]))
        print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
        errors.append(err_ref)

    print('\nOptimization complete.')
    image_final = mi.render(scene, spp=128)
    mi.util.write_bitmap("img_final.png", image_final)

    plt.plot(errors)
    plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
    plt.savefig("img_loss.png")

if __name__ == "__main__":
    main()  




