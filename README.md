# Evaluating Invertible Architectures for Koopman-Based Prediction of Frequency Dynamics in Power Grids
Official implementation of the paper  
**"Evaluating Invertible Architectures for Koopman-Based Prediction of Frequency Dynamics in Power Grids"**,  
by Eric Lupascu, Xiao Li, and Benjamin Schäfer, submitted to 2026 Open Source Modelling and Simulation of Energy System (OSMSES).

---

## 📝 Overview

This repository contains the official implementation of our paper **"Evaluating Invertible Architectures for Koopman-Based Prediction of Frequency Dynamics in Power Grids"**.  
The code reproduces the main experiments and figures presented in the paper.

If you find this repository useful, please consider citing our work (see the **Citation** section below).

---

## 📦 Requirements

The code was developed and tested with the following environment:

- Python >= 3.10  
- PyTorch >= 2.0  
- [List other key dependencies, e.g.:]  
  - gpytorch  
  - torchdiffeq  
  - numpy  
  - matplotlib  
  - scipy  

To install all dependencies:
```bash
pip install -r requirements.txt

🚀 Usage
1. Clone the repository
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]

2. Prepare data

[Describe how to download or prepare the dataset used in your paper, e.g.:]

mkdir data
# Place data files here or run the following to download automatically
python scripts/download_data.py

3. Run training / simulation

[Adapt to your code structure, e.g.:]

python train.py --config configs/experiment.yaml

4. Evaluate or visualize results
python evaluate.py --checkpoint checkpoints/model_final.pth
python plot_results.py


📚 Citation

If you use this code or find our work helpful, please cite:

@article{yourname2025paper,
  title={Your Paper Title},
  author={Your Name and Coauthor Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}

📄 License

This repository is released under the MIT License
.
You are free to use, modify, and distribute this code for research purposes, provided that proper credit is given.

🙏 Acknowledgments

We thank the contributors of the following open-source projects used in this work:

PyTorch

GPyTorch

torchdiffeq

[Add others as needed]

🧩 Contact

For questions or collaborations, please contact:

[Your Name], [Your Institution]

Email: [your_email@example.com
]
