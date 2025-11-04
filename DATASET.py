import numpy as np
import pandas as pd
from datetime import datetime

def generate_fea_dataset(n_samples=10000):
    """
    Generate synthetic FEA dataset based on theoretical formulas
    Using systematic variation (AP-like sequences) with 5 specific materials
    """
    
    # Define 5 specific materials with their properties
    materials = {
        'Aluminum_6061': {
            'youngs_modulus': 69000,  # MPa
            'poisson_ratio': 0.33,
            'yield_strength': 276,     # MPa
            'density': 2700            # kg/m³
        },
        'Steel_AISI_1020': {
            'youngs_modulus': 200000,
            'poisson_ratio': 0.29,
            'yield_strength': 350,
            'density': 7850
        },
        'Titanium_Ti6Al4V': {
            'youngs_modulus': 113800,
            'poisson_ratio': 0.342,
            'yield_strength': 880,
            'density': 4430
        },
        'Copper_C11000': {
            'youngs_modulus': 117000,
            'poisson_ratio': 0.355,
            'yield_strength': 220,
            'density': 8940
        },
        'Stainless_Steel_304': {
            'youngs_modulus': 193000,
            'poisson_ratio': 0.29,
            'yield_strength': 215,
            'density': 8000
        }
    }
    
    material_list = list(materials.keys())
    
    # Generate systematic ranges (Arithmetic Progression style)
    # Geometry parameters - evenly spaced
    lengths = np.linspace(100, 2000, 100)      # mm
    widths = np.linspace(10, 200, 50)          # mm
    heights = np.linspace(5, 100, 40)          # mm
    thicknesses = np.linspace(2, 50, 30)       # mm
    
    # Loading conditions - evenly spaced
    forces = np.linspace(100, 50000, 100)      # N
    moments = np.linspace(0, 100000, 100)      # N-mm
    pressures = np.linspace(0, 50, 50)         # MPa
    torques = np.linspace(0, 50000, 100)       # N-mm
    
    # Load types and boundary conditions
    load_types = [0, 1, 2]  # 0: tensile, 1: bending, 2: combined
    boundary_conditions = [0, 1, 2]  # 0: fixed-free, 1: fixed-fixed, 2: pinned-pinned
    
    data = []
    
    for i in range(n_samples):
        # Select systematically from arrays
        length = lengths[i % len(lengths)]
        width = widths[i % len(widths)]
        height = heights[i % len(heights)]
        thickness = thicknesses[i % len(thicknesses)]
        
        # Select material systematically
        material_name = material_list[i % len(material_list)]
        material_props = materials[material_name]
        
        youngs_modulus = material_props['youngs_modulus']
        poisson_ratio = material_props['poisson_ratio']
        yield_strength = material_props['yield_strength']
        density = material_props['density']
        
        # Select loads systematically
        force = forces[i % len(forces)]
        moment = moments[i % len(moments)]
        pressure = pressures[i % len(pressures)]
        torque = torques[i % len(torques)]
        
        load_type = load_types[i % len(load_types)]
        boundary_condition = boundary_conditions[i % len(boundary_conditions)]
        
        # Calculate cross-sectional properties
        area = width * thickness  # mm²
        I = (width * height**3) / 12  # Second moment of area (mm⁴)
        J = (width * height * (width**2 + height**2)) / 12  # Polar moment (mm⁴)
        
        # Calculate stresses using theoretical formulas
        
        # 1. Axial stress (σ = F/A)
        axial_stress = force / area if area > 0 else 0
        
        # 2. Bending stress (σ = M*c/I)
        c = height / 2  # Distance to neutral axis
        bending_stress = (moment * c) / I if I > 0 else 0
        
        # 3. Shear stress (τ = V*Q/(I*b))
        # Simplified: τ_max = 1.5 * V / A for rectangular section
        shear_stress = 1.5 * force / area if area > 0 else 0
        
        # 4. Torsional stress (τ = T*r/J)
        r = min(width, height) / 2
        torsional_stress = (torque * r) / J if J > 0 else 0
        
        # 5. Direct pressure stress
        pressure_stress = pressure
        
        # Combined stress based on load type
        if load_type == 0:  # Tensile
            combined_stress = axial_stress + pressure_stress
        elif load_type == 1:  # Bending
            combined_stress = bending_stress + shear_stress * 0.5
        else:  # Combined loading
            combined_stress = np.sqrt(
                (axial_stress + bending_stress)**2 + 
                3 * (shear_stress + torsional_stress)**2
            )
        
        # Von Mises stress (more realistic for ductile materials)
        von_mises_stress = np.sqrt(
            combined_stress**2 + 
            3 * (shear_stress + torsional_stress)**2
        )
        
        # Maximum stress (target variable)
        max_stress = max(combined_stress, von_mises_stress)
        
        # Add small systematic variation (±2%) to avoid exact duplicates
        variation = 1 + 0.02 * np.sin(i * 0.1)
        max_stress *= variation
        
        # Calculate safety factor
        safety_factor = yield_strength / max_stress if max_stress > 0 else 100
        
        # Calculate deflection (simplified for cantilever beam)
        deflection = (force * length**3) / (3 * youngs_modulus * I) if I > 0 else 0
        deflection *= variation
        
        # Strain
        strain = max_stress / youngs_modulus if youngs_modulus > 0 else 0
        
        data.append({
            # Geometry
            'length_mm': round(length, 2),
            'width_mm': round(width, 2),
            'height_mm': round(height, 2),
            'thickness_mm': round(thickness, 2),
            'area_mm2': round(area, 2),
            'moment_of_inertia_mm4': round(I, 2),
            
            # Material properties
            'material_name': material_name,
            'youngs_modulus_MPa': youngs_modulus,
            'poisson_ratio': poisson_ratio,
            'yield_strength_MPa': yield_strength,
            'density_kg_m3': density,
            
            # Loading
            'force_N': round(force, 2),
            'moment_Nmm': round(moment, 2),
            'pressure_MPa': round(pressure, 2),
            'torque_Nmm': round(torque, 2),
            'load_type': load_type,
            'boundary_condition': boundary_condition,
            
            # Results (targets)
            'max_stress_MPa': round(max_stress, 2),
            'von_mises_stress_MPa': round(von_mises_stress, 2),
            'safety_factor': round(safety_factor, 2),
            'deflection_mm': round(deflection, 4),
            'strain': round(strain, 6)
        })
    
    return pd.DataFrame(data)

# Generate dataset
print("=" * 70)
print("FEA DATASET GENERATOR - SYSTEMATIC APPROACH")
print("=" * 70)
print("\nGenerating FEA dataset with 10,000 samples...")
print("Using 5 standard materials with systematic parameter variation\n")

print("Materials included:")
print("  1. Aluminum 6061-T6")
print("  2. Steel AISI 1020")
print("  3. Titanium Ti-6Al-4V")
print("  4. Copper C11000")
print("  5. Stainless Steel 304")
print("\nGenerating...")

df = generate_fea_dataset(10000)

# Save to CSV
filename = f'fea_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(filename, index=False)

print(f"\n✓ Dataset generated successfully!")
print(f"✓ Saved as: {filename}")
print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nMaterial distribution:")
print(df['material_name'].value_counts())
print(f"\nLoad type distribution:")
load_type_map = {0: 'Tensile', 1: 'Bending', 2: 'Combined'}
print(df['load_type'].map(load_type_map).value_counts())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget variable (max_stress_MPa) statistics:")
print(f"  Mean: {df['max_stress_MPa'].mean():.2f} MPa")
print(f"  Std:  {df['max_stress_MPa'].std():.2f} MPa")
print(f"  Min:  {df['max_stress_MPa'].min():.2f} MPa")
print(f"  Max:  {df['max_stress_MPa'].max():.2f} MPa")
print("\n" + "=" * 70)
