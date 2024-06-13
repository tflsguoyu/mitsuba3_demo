import mitsuba as mi
mi.set_variant("scalar_rgb")

scene = mi.load_file("xml/cbox.xml")
image = mi.render(scene, spp=256)
mi.util.write_bitmap("cbox.png", image)
