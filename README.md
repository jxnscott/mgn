# mgn

A from-scratch PyTorch reimplementation of **MeshGraphNets** (Pfaff et al., 2021,
*Learning Mesh-Based Simulation with Graph Networks*), built for understanding and
extension.

## Setup

````bash
uv sync --all-groups          # dev
uv sync --group core_cpu      # runtime
````

## Reference

````bibtex
@article{DBLP:journals/corr/abs-2010-03409,
  author       = {Tobias Pfaff and
                  Meire Fortunato and
                  Alvaro Sanchez{-}Gonzalez and
                  Peter W. Battaglia},
  title        = {Learning Mesh-Based Simulation with Graph Networks},
  journal      = {CoRR},
  volume       = {abs/2010.03409},
  year         = {2020},
  url          = {https://arxiv.org/abs/2010.03409},
  eprinttype   = {arXiv},
  eprint       = {2010.03409}
}
````