import matplotlib.pyplot as plt
plt.plot(); plt.show()
import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import mitransient as mitr
import imageio
dr.set_flag(dr.JitFlag.Default, True)

# Modified from: https://mitsuba.readthedocs.io/en/latest/src/others/bsdf_deep_dive.html

def sph_to_dir(theta, phi):
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

if __name__=='__main__':
    debug_bsdfs = False # Set to true to get plots for the data measurements and fits

    cardboard_400 = mi.load_dict({
        'type': 'diffuse_swir',
        'csv_file': 'measurements/400/ceramic.csv',
        'gaussian': False,
        'debug': debug_bsdfs
    })

    cardboard_600 = mi.load_dict({
        'type': 'diffuse_swir',
        'csv_file': 'measurements/600/ceramic.csv',
        'gaussian': False,
        'debug': debug_bsdfs
    })

    cardboard_1550 = mi.load_dict({
        'type': 'diffuse_swir',
        'csv_file': 'measurements/1550/ceramic.csv',
        'gaussian': False,
        'debug': debug_bsdfs
    })

    # Create a (dummy) surface interaction to use for the evaluation of the BSDF
    si = dr.zeros(mi.SurfaceInteraction3f)

    # Specify an incident direction with 50 degrees elevation
    si.wi = sph_to_dir(dr.deg2rad(50.0), 0.0)

    # Create grid in spherical coordinates and map it onto the sphere
    # Theta Range = 0, pi
    # Phi Range = 0, 2pi
    res = 300
    theta_o, phi_o = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.pi, res),
        dr.linspace(mi.Float, 0, 2 * dr.pi, res)
    )
    theta_o_array = np.array(theta_o)
    phi_o_array = np.array(phi_o)
    wo = sph_to_dir(theta_o, phi_o)
    wo_array = np.array(wo)

    # Evaluate the whole array at once
    values_cardboard_400 = mi.Color3f(cardboard_400.eval(mi.BSDFContext(), si, wo))
    values_cardboard_600 = mi.Color3f(cardboard_600.eval(mi.BSDFContext(), si, wo))
    values_cardboard_1550 = mi.Color3f(cardboard_1550.eval(mi.BSDFContext(), si, wo))

    # Extract red channel of BRDF values and reshape into 2D grid
    values_cardboard_400_np = np.array(values_cardboard_400)
    values_cardboard_400_np_r = values_cardboard_400_np[0, :]
    values_cardboard_400_np_r = values_cardboard_400_np_r.reshape(res, res).T

    values_cardboard_600_np = np.array(values_cardboard_600)
    values_cardboard_600_np_r = values_cardboard_600_np[0, :]
    values_cardboard_600_np_r = values_cardboard_600_np_r.reshape(res, res).T

    values_cardboard_1550_np = np.array(values_cardboard_1550)
    values_cardboard_1550_np_r = values_cardboard_1550_np[0, :]
    values_cardboard_1550_np_r = values_cardboard_1550_np_r.reshape(res, res).T

    # Plot values for spherical coordinates
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    im = ax1.imshow(values_cardboard_400_np_r, extent=[0, 2 * np.pi, np.pi, 0], cmap='jet')

    ax1.set_xlabel(r'$\phi_o$', size=10)
    ax1.set_xticks([0, dr.pi, dr.two_pi])
    ax1.set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
    ax1.set_ylabel(r'$\theta_o$', size=10)
    ax1.set_yticks([0, dr.pi / 2, dr.pi])
    ax1.set_yticklabels(['-$\\pi/2$', '0', '$\\pi/2$'])
    ax1.set_title(r'$\lambda$ = 400nm')

    im = ax2.imshow(values_cardboard_600_np_r, extent=[0, 2 * np.pi, np.pi, 0], cmap='jet')

    ax2.set_xlabel(r'$\phi_o$', size=10)
    ax2.set_xticks([0, dr.pi, dr.two_pi])
    ax2.set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
    ax2.set_ylabel(r'$\theta_o$', size=10)
    ax2.set_yticks([0, dr.pi / 2, dr.pi])
    ax2.set_yticklabels(['-$\\pi/2$', '0', '$\\pi/2$'])
    ax2.set_title(r'$\lambda$ = 600nm')

    im = ax3.imshow(values_cardboard_1550_np_r, extent=[0, 2 * np.pi, np.pi, 0], cmap='jet')

    ax3.set_xlabel(r'$\phi_o$', size=10)
    ax3.set_xticks([0, dr.pi, dr.two_pi])
    ax3.set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
    ax3.set_ylabel(r'$\theta_o$', size=10)
    ax3.set_yticks([0, dr.pi / 2, dr.pi])
    ax3.set_yticklabels(['-$\\pi/2$', '0', '$\\pi/2$'])
    ax3.set_title(r'$\lambda$ = 1550nm')

    plt.show()

    # Evaluate modifying just the theta angle
    thetas = dr.linspace(mi.Float, -dr.pi / 2, dr.pi / 2, res)
    wo = sph_to_dir(thetas, 0.0)
    
    values_cardboard_400 = cardboard_400.eval(mi.BSDFContext(), si, wo)
    values_cardboard_400_np = np.array(values_cardboard_400)

    values_cardboard_600 = cardboard_600.eval(mi.BSDFContext(), si, wo)
    values_cardboard_600_np = np.array(values_cardboard_600)

    values_cardboard_1550 = cardboard_1550.eval(mi.BSDFContext(), si, wo)
    values_cardboard_1550_np = np.array(values_cardboard_1550)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.plot(np.array(thetas), values_cardboard_400_np, label='Measured reflectance @ 400nm')
    plt.plot(np.array(thetas), values_cardboard_600_np, label='Measured reflectance @ 600nm')
    plt.plot(np.array(thetas), values_cardboard_1550_np, label='Measured reflectance @ 1550nm')
    ax = plt.gca()
    ax.set_ylabel('Reflectance', size=10)
    ax.set_xlabel(r'$\theta_o$ (Elevation)', size=10)
    ax.set_xticks([-dr.pi / 2, 0, dr.pi / 2])
    ax.set_xticklabels(['-$\\pi/2$', '0', '$\\pi/2$'])
    plt.title('Measured reflectance given wavelength')
    plt.legend()
    plt.show()
    
    # Try to render a scene with the bsdf
    scene = mi.load_file('Scenes/steady_scene.xml')
    img = np.array(mi.render(scene=scene, spp=100))
    imageio.imwrite('render_gauss.png', img ** (1.0/2.2))
    plt.axis("off")
    plt.imshow(img ** (1.0/2.2))
    plt.show()
