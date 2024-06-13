import mitsuba as mi  # pip install mitsuba
mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T

def scene_dict():
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'sensor': {
            'type': 'perspective',
            'fov': 60,
            'to_world': T.look_at(
                target=[0, 0, 0],
                origin=[3, 0, 0],
                up=[0, 0, 1]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 1
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 1024,
                'rfilter': {
                    'type': 'box'
                },
                'pixel_format': 'rgb'
            }
        },
        'shape1': {
            'type': 'obj',
            'to_world': mi.Transform4f.scale([0.01, 0.01, 0.01]).translate([0, 0, -120]).rotate(axis=[1, 0, 0], angle=90),
            'filename': 'meshes/globe1.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'maps/world.jpg',
                    'raw': False
                }
            }
        },
        'shape2': {
            'type': 'obj',
            'to_world': mi.Transform4f.scale([0.01, 0.01, 0.01]).translate([0, 0, -120]).rotate(axis=[1, 0, 0], angle=90),
            'filename': 'meshes/globe2.obj',
            'bsdf': {
                'type': 'roughconductor',
                'material': 'Au',
                'distribution': "ggx"
            }
        },
        'emitter': {
            'type': 'point',
            'position': [3, -3, 3],
            'intensity': {
                'type': 'spectrum',
                'value': 100.0,
            }
        },
        'emitter_const': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 0.1
            }
        }
    }

    return scene


def main():
    scene = mi.load_dict(scene_dict())
    image = mi.render(scene)
    mi.util.write_bitmap("globe.png", image)
 

if __name__ == "__main__":
    main()  

