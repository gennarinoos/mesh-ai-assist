import os

import trimesh
from trimesh import Scene, Trimesh

from neural_mesh_simplification import NeuralMeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()


def save_mesh_to_file(mesh: trimesh.Geometry, file_name: str):
    """
    Save the simplified mesh to file in the simplified folder.
    """
    simplified_dir = os.path.join(data_dir, "processed")
    os.makedirs(simplified_dir, exist_ok=True)
    output_path = os.path.join(simplified_dir, file_name)
    mesh.export(output_path)

    print(f"Mesh saved to: {output_path}")


def cube_example():
    print(f"Creating cube mesh")
    file = "cube.obj"
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    simplified_mesh = simplifier.simplify(mesh)
    save_mesh_to_file(mesh, file)
    save_mesh_to_file(simplified_mesh, f"simplified_{file}")


def sphere_example():
    print(f"Creating sphere mesh")
    file = "sphere.obj"
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2)
    simplified_mesh = simplifier.simplify(mesh)
    save_mesh_to_file(mesh, file)
    save_mesh_to_file(simplified_mesh, f"simplified_{file}")


def cylinder_example():
    print(f"Creating cylinder mesh")
    file = "cylinder.obj"
    mesh = trimesh.creation.cylinder(radius=1, height=2)
    simplified_mesh = simplifier.simplify(mesh)
    save_mesh_to_file(mesh, file)
    save_mesh_to_file(simplified_mesh, f"simplified_{file}")


def mesh_dropbox_example():
    print(f"Loading all meshes of type '.obj' in folder '{data_dir}'")
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
        save_mesh_to_file(simplified_scene, f"simplified_{file_name}")


def main():
    # cube_example()
    # sphere_example()
    # cylinder_example()
    mesh_dropbox_example()


if __name__ == "__main__":
    main()
