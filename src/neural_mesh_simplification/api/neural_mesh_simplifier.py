import torch
from torch_geometric.data import Data
import trimesh
from ..models import NeuralMeshSimplification
from ..data.dataset import mesh_to_tensor, preprocess_mesh

class NeuralMeshSimplifier():
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, k=5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.model = self._build_model()

    def _build_model(self):
        return NeuralMeshSimplification(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            k=self.k
        )

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:

        # Preprocess the mesh (e.g. normalize, center)
        # preprocesed_mesh: trimesh.Trimesh = preprocess_mesh(mesh)
        preprocesed_mesh = mesh

        # Convert to a tensor
        tensor: Data = mesh_to_tensor(preprocesed_mesh)
        print(tensor)
        # model_output = self.model(tensor)

        # x = tensor.x
        # edge_index = tensor.edge_index
        # input_data = Data(x=x, edge_index=edge_index)

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