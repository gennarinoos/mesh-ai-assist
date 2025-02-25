import pytest
import torch
from torch_geometric.data import Data

from neural_mesh_simplification.models import NeuralMeshSimplification


@pytest.fixture
def sample_data() -> Data:
    x = torch.randn(10, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    pos = torch.randn(10, 3)
    return Data(x=x, edge_index=edge_index, pos=pos)


def test_neural_mesh_simplification_forward(sample_data: Data):
    model = NeuralMeshSimplification(input_dim=3, hidden_dim=64, num_layers=3, k=5)
    output = model(sample_data)

    # Add assertions to check the output structure and shapes
    assert isinstance(output, dict)
    assert "sampled_indices" in output
    assert "sampled_probs" in output
    assert "sampled_vertices" in output  # New check
    assert "edge_index" in output
    assert "edge_probs" in output
    assert "candidate_triangles" in output
    assert "triangle_probs" in output
    assert "face_probs" in output
    assert "simplified_faces" in output  # New check

    # Check shapes
    assert output["sampled_indices"].dim() == 1
    assert output["sampled_probs"].shape == (10,)
    assert output["sampled_vertices"].dim() == 2
    assert output["sampled_vertices"].shape[1] == 3
    assert output["edge_index"].shape[0] == 2
    assert output["edge_probs"].dim() == 1
    assert (
        output["candidate_triangles"].shape[1] == 3
        if output["candidate_triangles"].numel() > 0
        else True
    )
    assert output["triangle_probs"].dim() == 1
    assert (
            output["face_probs"].shape[0] == output["candidate_triangles"].shape[0]
    )  # New assertion
    assert (
        output["simplified_faces"].shape[1] == 3
        if output["simplified_faces"].numel() > 0
        else True
    )

    # Additional checks
    assert (
            output["sampled_indices"].shape[0] <= 10
    )  # number of sampled points should be less than or equal to input
    assert (
            output["sampled_vertices"].shape[0] == output["sampled_indices"].shape[0]
    )  # New check
    assert (
            output["edge_index"].shape[1] == output["edge_probs"].shape[0]
    )  # number of edges should match number of edge probabilities
    assert (
            output["candidate_triangles"].shape[0] == output["triangle_probs"].shape[0]
    )  # number of candidate triangles should match number of triangle probabilities
    assert (
            output["simplified_faces"].shape[0] <= output["candidate_triangles"].shape[0]
    )  # New check

    # Check that sampled_vertices are a subset of the original vertices
    original_vertices = sample_data.pos
    assert torch.all(
        torch.isin(output["sampled_vertices"], original_vertices)
    )  # New check

    # Check that simplified_faces only contain valid indices
    max_index = output["sampled_vertices"].shape[0] - 1
    assert torch.all(output["simplified_faces"] >= 0) and torch.all(
        output["simplified_faces"] <= max_index
    )  # New check

    # Check that simplified_faces only contain valid indices if not empty
    if output["simplified_faces"].numel() > 0:
        max_index = output["sampled_vertices"].shape[0] - 1
        assert torch.all(output["simplified_faces"] >= 0) and torch.all(
            output["simplified_faces"] <= max_index
        )

    # Check the relationship between face_probs and simplified_faces
    assert output["simplified_faces"].shape[0] <= output["face_probs"].shape[0]


def test_generate_candidate_triangles():
    model = NeuralMeshSimplification(input_dim=3, hidden_dim=64, num_layers=3, k=5)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    edge_probs = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.7, 0.7])

    triangles, triangle_probs = model.generate_candidate_triangles(
        edge_index, edge_probs
    )

    assert triangles.shape[1] == 3
    assert triangle_probs.shape[0] == triangles.shape[0]
    assert torch.all(triangles >= 0)
    assert torch.all(triangles < edge_index.max() + 1)
    assert torch.all(triangle_probs >= 0) and torch.all(triangle_probs <= 1)

    max_possible_triangles = edge_index.max().item() + 1  # num_nodes
    max_possible_triangles = (
            max_possible_triangles
            * (max_possible_triangles - 1)
            * (max_possible_triangles - 2)
            // 6
    )
    assert triangles.shape[0] <= max_possible_triangles
