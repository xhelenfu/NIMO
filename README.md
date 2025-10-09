# NIMO (Niches and biomarkers of IMmunOtherapy resistance)

For more details, please refer to our paper [Spatial transcriptomics reveals distinct cellular niches of resistance to immune checkpoint blockade in melanoma](TODO add link).

This repository contains the code for NIMO. Leveraging single-cell spatially resolved transcriptomics data, NIMO models local cellular neighbourhoods to identify recurring spatial patterns (niches) and predict patient response to immune checkpoint blockade.

![alt text](Figure1.png)

## Installation

> **Note**: A GPU is strongly recommended for the deep learning component.

1. Clone repository:
```sh
git clone https://github.com/xhelenfu/NIMO.git
```

2. Create virtual environment:
```sh
conda create --name nimo python=3.12
```

3. Activate virtual environment:
```sh
conda activate nimo
```

4. Install dependencies:
```sh
cd nimo
pip install -r requirements.txt
```
Installation is expected to be completed within a few minutes.


## Tutorials

Please check out [tutorials](./tutorials/) for key use cases of NIMO, including:

- [Training and validation](./tutorials/1_training_and_validation.ipynb)


## Citation

If NIMO has assisted you with your work, please kindly cite our paper: TODO add link