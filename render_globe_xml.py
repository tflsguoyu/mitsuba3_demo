import mitsuba as mi
mi.set_variant("scalar_rgb")

scene = mi.load_file("xml/globe.xml")
image = mi.render(scene)
mi.util.write_bitmap("globe_xml_path.png", image)
