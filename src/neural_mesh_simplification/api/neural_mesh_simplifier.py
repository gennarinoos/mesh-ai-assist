from ..models import NeuralMeshSimplification

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

    def simplify(self, mesh):
        return self.model.simplify(mesh)