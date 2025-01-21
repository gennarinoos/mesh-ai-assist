# import sys
# print(sys.path)

import os
from neural_mesh_simplification import NeuralMeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh
from neural_mesh_simplification.utils import save_mesh

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()

# Load a mesh (replace with a path to your .obj file)
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(script_dir, "data", "quit.obj")
original_mesh = load_mesh(mesh_path)

# Simplify the mesh
simplified_mesh = simplifier.simplify(original_mesh)

# Save the simplified mesh
output_path = os.path.join(script_dir, "data", "simplified_quit.obj")
save_mesh(simplified_mesh, output_path)

print(f"Simplified mesh saved to: {output_path}")
