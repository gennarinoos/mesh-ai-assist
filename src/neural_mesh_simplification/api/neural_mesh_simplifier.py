import torch
import trimesh
from torch_geometric.data import Data

from ..data.dataset import mesh_to_tensor, preprocess_mesh
from ..models import NeuralMeshSimplification


class NeuralMeshSimplifier:
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, k=5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.model = self._build_model()

    @classmethod
    def using_model(cls, at_path: str, map_location: str):
        instance = cls()
        instance._load_model(at_path, map_location)
        return instance

    def _build_model(self):
        return NeuralMeshSimplification(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            k=self.k
        )

    def _load_model(self, checkpoint_path: str, map_location: str):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    def _simplify_1(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        vertices, faces = mesh.vertices, mesh.faces
        vertices = torch.tensor(vertices, dtype=torch.float32, device="cpu").unsqueeze(0)  # Shape: [1, N, 3]

        # Simplify the mesh
        print("Simplifying the mesh...")
        with torch.no_grad():
            simplified_vertices = self.model(vertices)  # Model should output simplified vertices

        simplified_vertices = simplified_vertices.squeeze(0).cpu().numpy()  # Shape: [N', 3]

        return trimesh.Trimesh(vertices=simplified_vertices)

    def _simplify_2(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        # Preprocess the mesh (e.g. normalize, center)
        preprocesed_mesh: trimesh.Trimesh = preprocess_mesh(mesh)

        # Convert to a tensor
        tensor: Data = mesh_to_tensor(preprocesed_mesh)
        model_output = self.model(tensor)

        vertices = model_output["sampled_vertices"].detach().numpy()
        faces = model_output["simplified_faces"].numpy()
        edges = model_output["edge_index"].t().numpy()  # Transpose to get (n, 2) shape

        return trimesh.Trimesh(vertices=vertices, faces=faces, edges=edges)

    def _dummy_simplify(self) -> trimesh.Trimesh:
        x = torch.randn(10, 3)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
        pos = torch.randn(10, 3)
        input_data = Data(x=x, edge_index=edge_index, pos=pos)

        model_output = self.model(input_data)

        # Convert the model output into a mesh and return it
        vertices = model_output["sampled_vertices"].detach().numpy()
        faces = model_output["simplified_faces"].numpy()
        edges = model_output["edge_index"].t().numpy()  # Transpose to get (n, 2) shape

        return trimesh.Trimesh(vertices=vertices, faces=faces, edges=edges)

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        # return self._dummy_simplify()
        # return self._simplify_1(mesh)
        return self._simplify_2(mesh)
