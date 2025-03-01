from dataclasses import dataclass


@dataclass
class SimpleAerodynamicsParameters:
    mass_kg: float = 3.0
    wing_area_m2: float = 0.6
    CD: float = 0.08684
    CL: float = 1.0
    rho_kg_per_m3: float = 1.225
    g_m_per_s2: float = 9.81


@dataclass
class SimpleAerodynamicsConfig(SimpleAerodynamicsParameters):
    _target_: str = 'env.glider.aerodynamics.SimpleAerodynamics'
