import trimesh

def save_mesh(mesh1):
    # Export the mesh to OBJ format
    mesh1.export('output_file.obj')
