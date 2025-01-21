# Mesh AI Assist

A collection of AI tools to work with 3D Meshes.

* Neural Mesh Simplification

---

## Neural Mesh Simplification

From the paper "Neural Mesh Simplification" by Potamias et al. (CVPR 2022), this Python package provides a fast, learnable method for mesh simplification that generates simplified meshes in real-time.

Research, methodology introduced in the [Neural Mesh Simplification paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Potamias_Neural_Mesh_Simplification_CVPR_2022_paper.pdf), with the updated info shared in [supplementary material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Potamias_Neural_Mesh_Simplification_CVPR_2022_supplemental.pdf).


### Overview

Neural Mesh Simplification is a novel approach to reduce the resolution of 3D meshes while preserving their appearance. Unlike traditional simplification methods that collapse edges in a greedy iterative manner, this method simplifies a given mesh in one pass using deep learning techniques.

The method consists of three main steps:

1. Sampling a subset of input vertices using a sophisticated extension of random sampling.
2. Training a sparse attention network to propose candidate triangles based on the edge connectivity of sampled vertices.
3. Using a classification network to estimate the probability that a candidate triangle will be included in the final mesh.

### Features

- Fast and scalable mesh simplification
- One-pass simplification process
- Preservation of mesh appearance
- Lightweight and differentiable implementation
- Suitable for integration into learnable pipelines

## Installation

```bash
conda create -n neural-mesh-simplification python=3.12
conda activate neural-mesh-simplification
conda install pip
pip install -r requirements.txt
pip install -e .
```

## Example Usage

1. Drop your meshes as `.obj` files to the `examples/data` folder
2. Run the following command
```bash
python examples/example.py
```
3. Collect the simplified meshes in `examples/data/simplified`


## Training

To train the model on your own dataset:

```bash
python ./scripts/train.py --data_path /path/to/your/dataset --epochs 100 --batch_size 32
```

## Evaluation

To evaluate the model on a test set:

```bash
python ./scripts/evaluate.py --model_path /path/to/saved/model --test_data /path/to/test/set
```

## Citation

If you use this code in your research, please cite the original paper:

```
@InProceedings{Potamias_2022_CVPR,
    author    = {Potamias, Rolandos Alexandros and Ploumpis, Stylianos and Zafeiriou, Stefanos},
    title     = {Neural Mesh Simplification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18583-18592}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
