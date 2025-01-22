import os

import trimesh
from trimesh import Scene, Trimesh

from neural_mesh_simplification import NeuralMeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()


def save_simplified_mesh(mesh: trimesh.Geometry, file_name: str):
    """
    Save the simplified mesh to file in the simplified folder.
    """
    simplified_dir = os.path.join(data_dir, "simplified")
    if not os.path.exists(simplified_dir):
        os.makedirs(simplified_dir)
    output_path = os.path.join(simplified_dir, f"simplified_{file_name}")
    mesh.export(output_path)

    print(f"Simplified mesh saved to: {output_path}")


def cubeExample():
    file = "cube.obj"
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    output_path = os.path.join(data_dir, file)
    mesh.export(output_path)
    simplified_mesh = simplifier.simplify(mesh)
    save_simplified_mesh(simplified_mesh, file)


def sphereExample():
    file = "sphere.obj"
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2)
    output_path = os.path.join(data_dir, file)
    mesh.export(output_path)
    simplified_mesh = simplifier.simplify(mesh)
    save_simplified_mesh(simplified_mesh, file)


def cylinderExample():
    file = "cylinder.obj"
    mesh = trimesh.creation.cylinder(radius=1, height=2)
    output_path = os.path.join(data_dir, file)
    mesh.export(output_path)
    simplified_mesh = simplifier.simplify(mesh)
    save_simplified_mesh(simplified_mesh, file)


def meshDropboxExample():
    # Load all meshes in the data folder (replace with a path to your .obj file)
    mesh_files = [f for f in os.listdir(data_dir) if f.endswith('.obj')]

    for file_name in mesh_files:
        mesh_path = os.path.join(data_dir, file_name)

        original_mesh = load_mesh(mesh_path)

        print("Loaded mesh at file" + mesh_path)

        # Create a new scene to hold the simplified meshes
        simplified_scene = Scene()

        if isinstance(original_mesh, Trimesh):
            print("Original: ", original_mesh.vertices.shape, original_mesh.edges.shape, original_mesh.faces.shape)
            simplified_geom = simplifier.simplify(original_mesh)
            print("Simplified: ", simplified_geom.vertices.shape, simplified_geom.edges.shape,
                  simplified_geom.faces.shape)

            simplified_scene = simplified_geom

        elif isinstance(original_mesh, Scene):
            # Iterate through the original mesh geometry
            for name, geom in original_mesh.geometry.items():
                print("Original: ", geom)
                # Simplify each Trimesh object
                simplified_geom = simplifier.simplify(geom)
                print("Simplified: ", simplified_geom)
                # Add the simplified geometry to the new scene
                simplified_scene.add_geometry(simplified_geom, geom_name=name)
        else:
            raise ValueError("Invalid mesh type (expected Trimesh or Scene):", type(original_mesh))

        # Save the simplified mesh to file
        save_simplified_mesh(simplified_scene, file_name)


def main():
    cubeExample()
    # sphereExample()
    # cylinderExample()
    # meshDropboxExample()


if __name__ == "__main__":
    main()
