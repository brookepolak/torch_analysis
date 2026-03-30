import yt
import numpy as np

def get_roi_region(ds, parfile="flash.par"):
    """
    Return yt region corresponding to derefinement ROI.
    """
    from torch_param import FlashPar  
    flashp = FlashPar(parfile)

    left_edge = ds.arr(
        [
            flashp["deref_xl"],
            flashp["deref_yl"],
            flashp["deref_zl"],
        ],
        "cm",
    )

    right_edge = ds.arr(
        [
            flashp["deref_xr"],
            flashp["deref_yr"],
            flashp["deref_zr"],
        ],
        "cm",
    )

    center = 0.5 * (left_edge + right_edge)

    return ds.region(
        center=center,
        left_edge=left_edge,
        right_edge=right_edge,
    )

def gas_mass_container(container):
    return (
        container[("gas", "mass")]
        .sum()
        .to("Msun")
        .value
    )


def gas_virial_ratio_container(container):
    dens = container[('flash','dens')]
    vx = container[('flash','velx')]
    vy = container[('flash','vely')]
    vz = container[('flash','velz')]
    v2 = vx**2 + vy**2 + vz**2

    Ekin = 0.5*dens*v2
    # Potential energy from stars and gas+sinks
    gas_gas_gpot = container[('flash','gpot')]
    Epot = dens*gas_gas_gpot.v

    alpha = abs(sum(Ekin.v)/sum(Epot.v))

    return alpha

def bound_gas_mass_fraction_container(container):
    mass = container[('gas','mass')]
    dens = container[('flash','dens')]
    vx = container[('flash','velx')]
    vy = container[('flash','vely')]
    vz = container[('flash','velz')]
    v2 = vx**2 + vy**2 + vz**2

    Bx = container[('flash','magx')]
    By = container[('flash','magy')]
    Bz = container[('flash','magz')]
    B2 = Bx**2 + By**2 + Bz**2

    # Magnetic energy
    Emag = 0.5*B2
    # Kinetic energy
    Ekin = 0.5*dens*v2
    # Potential energy from stars and gas+sinks
    gas_gas_gpot = container[('flash','gpot')]
    star_gas_gpot = container[('flash','bgpt')]
    Epot = dens*(gas_gas_gpot.v+star_gas_gpot.v)
    # Thermal energy TODO: generalize gamma
    Ethe = 3.0/2.0*container[("flash","pres")]
    # Total energy
    Etot = (Ekin.v+Epot.v+Emag.v+Ethe.v)

    bound_idx = np.where(Etot < 0.0)
    
    bmf = sum(mass[bound_idx])/sum(mass)

    return bmf

def particle_mass_container(ds, particle_type="all"):
    """
    Returns the total particle mass in the dataset for the given particle type.
    
    Parameters
    ----------
    ds : yt Dataset
        The dataset to analyze.
    particle_type : str
        Type of particle to sum:
        - 'stars' : particle_csgm == 0
        - 'sinks' : particle_csgm != 0
        - 'all'   : all particles

    Returns
    -------
    float
        Total mass in Msun.
    """
    if not ds.particles_exist:
        return []

    ad = ds.all_data()

    pm = ad[("all", "particle_mass")].to("Msun").value

    if particle_type == "all":
        return pm.sum() if pm.size > 0 else 0.0

    # Get particle type mask
    csgm = ad[("all", "particle_csgm")].value
    if particle_type == "stars":
        mask = (csgm == 0)
    elif particle_type == "sinks":
        mask = (csgm != 0)
    else:
        raise ValueError(f"Unknown particle_type '{particle_type}'")

    if mask.sum() == 0:
        return 0.0

    return pm[mask]

# -----------------------------
# ROI-specific quantity functions
# -----------------------------
def gas_mass_roi(ds):
    region = get_roi_region(ds)
    return gas_mass_container(region)

def bound_gas_mass_fraction_roi(ds):
    region = get_roi_region(ds)
    return bound_gas_mass_fraction_container(region)

def gas_virial_ratio_roi(ds):
    region = get_roi_region(ds)
    return gas_virial_ratio_container(region)

def sfe_roi(ds):
    region = get_roi_region(ds)
    mstars = stellar_mass(ds)
    mgas = gas_mass_roi(ds)
    return mstars / mgas if mgas > 0.0 else 0.0

def gas_mass(ds):
    return gas_mass_container(ds.all_data())

def bound_gas_mass_fraction(ds):
    return bound_gas_mass_fraction_container(ds.all_data())

def gas_virial_ratio(ds):
    return gas_virial_ratio_container(ds.all_data())

def sink_mass(ds):
    if not ds.particles_exist:
        return 0.0
    else:
        return particle_mass_container(ds, particle_type="sinks").sum()

def stellar_mass(ds):
    if not ds.particles_exist:
        return 0.0
    else:
        return particle_mass_container(ds, particle_type="stars").sum()

def max_star_mass(ds):
    sm = particle_mass_container(ds, particle_type="stars")
    return max(sm, default=0)

def number_stars(ds):
    sm = particle_mass_container(ds, particle_type="stars")
    return len(sm)

def number_feedback_stars(ds):
    from amuse.units import units
    from torch_user import user_parameters
    p = user_parameters()
    sm = np.asarray(particle_mass_container(ds, particle_type="stars"))
    return len(sm[sm >= p['min_feedback_mass'].value_in(units.MSun)])

def number_sinks(ds):
    sm = sink_mass(ds)
    if sm == 0.0:
        return 0
    return len(sm(ds))

def sfe(ds):
    ms = stellar_mass(ds)
    mg = gas_mass(ds)
    return ms / mg if mg > 0 else 0.0

def sfr(ds, prev_values):
    if prev_values == None:
        # there are no previous values; return nan
        return np.nan

    prev_time, prev_star_mass = prev_values
    sm = stellar_mass(ds)
    dm = sm - prev_star_mass
    dt = (ds.current_time.in_units('Myr').v - prev_time)*1e6 # Myr to yr

    return dm/dt

def stellar_velocity_dispersion(ds):
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    vx = ad['all','particle_velx'][stars].v
    vy = ad['all','particle_vely'][stars].v
    vz = ad['all','particle_velz'][stars].v
    v2 = vx**2 + vy**2 + vz**2

    return np.std(np.sqrt(v2))

def half_mass_radius(ds):
    """
    Compute half-mass radius of the star cluster.
    """
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    # Convert positions to unitless cm
    px = ad[("all", "particle_position_x")][stars].to("pc").value
    py = ad[("all", "particle_position_y")][stars].to("pc").value
    pz = ad[("all", "particle_position_z")][stars].to("pc").value
    pm = ad[("all", "particle_mass")][stars].to("Msun").value

    tot_mass = np.sum(pm)
    comx = np.sum(pm*px)/tot_mass
    comy = np.sum(pm*py)/tot_mass
    comz = np.sum(pm*pz)/tot_mass

    r2 = abs((px-comx)**2 + (py-comy)**2 + (pz-comz)**2)

    sort_r = np.argsort(r2)

    m_sort = pm[sort_r]
    r2_sort = r2[sort_r]

    half_mass = tot_mass/2.0
    cum_mass = np.cumsum(m_sort)
    idx_hmr = np.argmax(cum_mass>half_mass)

    hmr = np.sqrt(r2_sort[idx_hmr])
    return hmr

def stellar_virial_ratio(ds):
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    pos = np.array([ad['all','particle_posx'][stars].v,
                               ad['all','particle_posy'][stars].v,
                               ad['all','particle_posz'][stars].v]).T

    m  = ad['all','particle_mass'][stars].v
    vx = ad['all','particle_velx'][stars].v
    vy = ad['all','particle_vely'][stars].v
    vz = ad['all','particle_velz'][stars].v
    v2 = vx**2 + vy**2 + vz**2

    Ekin = 0.5*m*v2
    Epot = m*ds.find_field_values_at_points('bgpt', pos*yt.units.cm).v

    return abs(sum(Ekin)/sum(Epot))

def unbound_star_ids(ds):
    """
    Returns the FLASH particle IDs of stars unbound to the system. 
    Expensive calculation (O(n^2)) due to the particle-particle potential calculation.
    """
    if not ds.particles_exist:
        return []
    from amuse.lab import Particles
    from amuse.units import units

    ad = ds.all_data()
    star_idx = ad['all', 'particle_csgm'] == 0.0

    mass = ad['all','particle_mass'][star_idx].v
    pos = np.array([ad['all','particle_posx'][star_idx].v, 
                               ad['all','particle_posy'][star_idx].v, 
                               ad['all','particle_posz'][star_idx].v]).T
    vel = np.array([ad['all','particle_velx'][star_idx].v,
                                    ad['all','particle_vely'][star_idx].v,
                                    ad['all','particle_velz'][star_idx].v]).T
    # Potential field of gas+sinks at star positions
    gas_star_gpot   = mass*ds.find_field_values_at_points('gpot',pos*yt.units.cm).v
    # Potential field of stars at star positions
    star_star_gpot = mass*ds.find_field_values_at_points('bgpt', pos*yt.units.cm).v

    # TODO: implement user option to use exact potential (sloooow) or grid potential (fast)
    if False:
        stars = Particles(len(mass))
        stars.mass     = mass | units.g
        stars.position = pos | units.cm
        stars.velocity = vel | units.cm/units.s

        # Potential field of stars 
        star_star_gpot = (stars.mass*stars.potential()).value_in(units.erg) 

    # Stellar kinetic energy
    v2 = np.linalg.norm(vel)
    Ekin = 0.5*mass*v2

    # Total energy of stars
    Etot = Ekin + star_star_gpot + gas_star_gpot

    # ID tags of stars
    tag = ad['all','particle_tag'][star_idx].v
    unbound_idx = np.where(Etot >= 0.0)
    runaways = tag[unbound_idx]
    n_run = len(runaways)
    print(f'I found {n_run} runaway stars!')

    return runaways

QUANTITY_REGISTRY = {
    # Global
    "gas_mass": gas_mass,
    "gas_virial_ratio": gas_virial_ratio,
    "stellar_mass": stellar_mass,
    "stellar_velocity_dispersion": stellar_velocity_dispersion,
    "stellar_virial_ratio": stellar_virial_ratio,
    "sink_mass": sink_mass,
    "sfe": sfe,
    "sfr": sfr,
    "half_mass_radius": half_mass_radius,
    "number_stars": number_stars,
    "number_sinks": number_sinks,
    "number_feedback_stars": number_feedback_stars,
    "max_star_mass": max_star_mass,
    "unbound_star_ids": unbound_star_ids,
    "bound_gas_mass_fraction": bound_gas_mass_fraction,

    # ROI
    "gas_mass_roi": gas_mass_roi,
    "gas_virial_ratio_roi": gas_virial_ratio_roi,
    "bound_gas_mass_fraction_roi": bound_gas_mass_fraction_roi,
    "sfe_roi": sfe_roi,
}

# Is the quantity scalar or vector? hdf5 needs to know
QUANTITY_TYPE = {
    # Global
    "gas_mass": 'scalar',
    "gas_virial_ratio": 'scalar',
    "stellar_mass":  'scalar',
    "stellar_velocity_dispersion":  'scalar',
    "stellar_virial_ratio": 'scalar',
    "sink_mass":  'scalar',
    "sfe":  'scalar',
    "sfr":  'scalar',
    "half_mass_radius":  'scalar',
    "number_stars":  'scalar',
    "number_sinks":  'scalar',
    "number_feedback_stars": 'scalar',
    "max_star_mass":  'scalar',
    "unbound_star_ids":  'vector',
    "bound_gas_mass_fraction":  'scalar',

    # ROI
    "gas_mass_roi":  'scalar',
    "gas_virial_ratio_roi": 'scalar',
    "bound_gas_mass_fraction_roi":  'scalar',
    "sfe_roi":  'scalar'
}

# -----------------------------
# Dictionary of nice plot labels
# -----------------------------
QUANTITY_LABELS = {
    "gas_mass": r"$M_\mathrm{gas}\,[M_\odot]$",
    "gas_virial_ratio": r"$\alpha_{\rm gas}$",
    "sink_mass": r"$M_\mathrm{sink}\,[M_\odot]$",
    "stellar_mass": r"$M_\star\,[M_\odot]$",
    "stellar_velocity_dispersion": r"$\sigma_v~[{\rm cm/s}]$",
    "stellar_virial_ratio": r"$\alpha_{\star}$",
    "sfe": r"$\epsilon_\star$",
    "sfr": r"$\rm{SFR}\,[\rm{M_\odot~yr^{-1}}]$",
    "half_mass_radius": r"$R_{1/2}~[\rm{pc}]$",
    "number_stars": r"$N_\star$",
    "number_sinks": r"$N_{\rm sink}$",
    "number_feedback_stars": r"$N_{\rm fb}$",
    "max_star_mass": r"$M_{\star,\rm max}\,[M_\odot]$",
    "bound_gas_mass_fraction": r"$M_\mathrm{gas, bound}/M_{\rm tot}$",
    

    "gas_mass_roi": r"$M_\mathrm{gas, ROI}\,[M_\odot]$",
    "gas_virial_ratio_roi": r"$\alpha_{\rm gas}$",
    "bound_gas_mass_fraction_roi": r"$M_\mathrm{gas, bound}/M_{\rm tot}$",
    "sfe_roi": r"$\epsilon_{\star,\ \mathrm{ROI}}$",
}

# Storing a list of quantity names from the hdf5 file
# that a quantity function needs from a previous timestep. 
# the only current such quantity is SFR, which needs the 
# time and stellar mass of the previous snapshot
QUANTITY_REQUISITES = {
    "sfr": ["time","stellar_mass"]
        }

