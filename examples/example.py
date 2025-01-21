# import sys
# print(sys.path)

import os
from trimesh.base import Scene, Trimesh

from neural_mesh_simplification import NeuralMeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh
from neural_mesh_simplification.utils import save_mesh

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()


# Load all meshes in the data folder (replace with a path to your .obj file)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

mesh_files = [f for f in os.listdir(data_dir) if f.endswith('.obj')]

for file_name in mesh_files:
    mesh_path = os.path.join(data_dir, file_name)

    original_mesh = load_mesh(mesh_path)

    print("Loaded mesh at file" + mesh_path)
    print(original_mesh)

    # Create a new scene to hold the simplified meshes
    simplified_scene = Scene()

    if isinstance(original_mesh, Trimesh):
        simplified_geom = simplifier.simplify(original_mesh)
        print(simplified_geom)

        simplified_scene = simplified_geom

    elif isinstance(original_mesh, Scene):
        # Iterate through the original mesh geometry
        for name, geom in original_mesh.geometry.items():
            # Simplify each Trimesh object
            simplified_geom = simplifier.simplify(geom)
            # Add the simplified geometry to the new scene
            simplified_scene.add_geometry(simplified_geom, geom_name=name)
    else:
        raise ValueError("Invalid mesh type (expected Trimesh or Scene):", type(original_mesh), type(Trimesh))

    # Save the simplified scene
    simplified_dir = os.path.join(data_dir, "simplified")
    if not os.path.exists(simplified_dir):
        os.makedirs(simplified_dir)
    output_path = os.path.join(simplified_dir, f"simplified_{file_name}")
    simplified_scene.export(output_path)

    print(f"Simplified mesh saved to: {output_path}")