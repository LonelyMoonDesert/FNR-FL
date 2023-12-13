![Banner](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20231211183533.png)

## FNR-FL

The official implementation of Feature Norm Regularized Federated Learning (FNR-FL) algorithm, which uniquely incorporates class average feature norms to enhance model accuracy and convergence in non-i.i.d. scenarios.

Paper: [Feature Norm Regularized Federated Learning: Transforming Skewed Distributions into Global Insights](https://arxiv.org/abs/2312.06951)



## Table of Contents

- [FNR-FL](#project-title)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
- [License](#license)

## Introduction to codes

```shell
│  config.py	
│  criteo-dis.npy
│  datasets.py             # Dataset-related
│  draw_noisy_image.py     # (Additional) Observe samples with added Gaussian noise, use `alg=fedavg` when calling
│  femnist-dis.npy
│  LICENSE_FNR-FL          # MIT LICENSE for this project (FNR-FL)
│  LICENSE_NIID-Bench      # MIT LICENSE for the base code of this project (NIID-Bench)
│  model.py                # Define some models
│  partition.py            # Define partitioning functions to achieve different non-i.i.d distributions
│  README.md
│  requirements.txt        # Project dependencies
│  resnetcifar.py          # Define ResNet
│  train.py                # (Main) Training entry point (for ResNet)
│  utils.py
│  vggmodel.py             # Define VGG
│        
├─models                    # Folder to store model files
│     celeba_model.py
│     mnist_model.py
│     svhn_model.py

```

## Experimental settings

|    **Non-i.i.d settings**     |                                                              |   **Partition**   | **Beta** | **Noise** |
| :---------------------------: | :----------------------------------------------------------: | :---------------: | :------: | :-------: |
|  **Label Distribution Skew**  |            **Distribution-based label imbalance**            |  noniid-labeldir  |   0.5    |     0     |
| **Feature Distribution Skew** |              **Noise-based feature imbalance**               |       homo        |    0     |    0.5    |
|       **Quantity Skew**       |                              /                               | iid-diff-quantity |   0.5    |     0     |
|        **Mixed skew**         |    **distribution-based label imbalance + quantity skew**    |       mixed       |    /     |     /     |
|        **Mixed skew**         | **distribution-based label imbalance + noise-based feature imbalance** |  noniid-labeldir  |   0.5    |    0.5    |

## Installation

[(Back to top)](#table-of-contents)

Install dependencies by running:

```shell
pip install -r requirements.txt
```



## Usage
[(Back to top)](#table-of-contents)

For example, to run the tests on ResNet18:

```shell
python train.py 
--model=resnet
--dataset=cifar10
--alg=fnr
--lr=0.01
--cc_optimizer=sgd
--batch-size=64
--test_batch_size=32
--epochs=10
--calibration_epochs=5
--n_parties=10
--mu=0.01
--rho=0.9
--comm_round=10
--partition=mixed
--noise=0.0
--beta=0.5
--device=cuda:0
--datadir=../data/
--logdir=./logs/
--sample=1
--init_seed=0
--ccreg_w=0.5
```

The parameters and their descriptions are listed as follows:

| Parameter            | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `model`              | The model architecture. Default = `resnet`.                  |
| `dataset`            | Dataset to use. Options: `mnist`, `cifar10`, `fmnist`, `svhn`. Default = `mnist`. |
| `alg`                | The training algorithm. Options: `fedavg`, `fedprox`, `scaffold`, `fednova`, `moon`,`fnr`. Default = `fnr`. |
| `lr`                 | Learning rate for the local models, default = `0.01`.        |
| `batch-size`         | Batch size, default = `64`.                                  |
| `epochs`             | Number of local training epochs, default = `10`.             |
| `calibration_epochs` | Number of calibration epochs, default = `5`.                 |
| `ccreg_w`            | Weight of calibration regularization term. Default=1.0       |
| `n_parties`          | Number of parties, default = `2`.                            |
| `mu`                 | The proximal term parameter for FedProx, default = `0.001`.  |
| `rho`                | The parameter controlling the momentum SGD, default = `0`.   |
| `comm_round`         | Number of communication rounds to use, default = `10`.       |
| `partition`          | The partition way. Options: `homo`, `noniid-labeldir`,  `iid-diff-quantity`. Default = `homo` |
| `beta`               | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device`             | Specify the device to run the program, default = `cuda:0`.   |
| `datadir`            | The path of the dataset, default = `./data/`.                |
| `logdir`             | The path to store the logs, default = `./logs/`.             |
| `noise`              | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample`             | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed`          | The initial seed, default = `0`.                             |

## Contribute
[(Back to top)](#table-of-contents)

Thanks goes to these wonderful people:

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LonelyMoonDesert"><img src="https://avatars.githubusercontent.com/u/56340292?v=4" width="100px;" alt="LonelyMoonDesert"/><br /><sub><b>LonelyMoonDesert</b></sub></a><br /><a href="https://github.com/LonelyMoonDesert/FNR-FL/commits?author=LonelyMoonDesert" title="Code">💻</a> <a href="" title="Design">🎨</a> <a href="" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Citation

If you find our work useful in your research, please consider citing our paper:

```shell
@misc{hu2023feature,
      title={Feature Norm Regularized Federated Learning: Transforming Skewed Distributions into Global Insights}, 
      author={Ke Hu and WeiDong Qiu and Peng Tang},
      year={2023},
      eprint={2312.06951},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



## Acknowledgments

We would like to express our sincere gratitude to the [Xtra-Computing Group](https://github.com/Xtra-Computing) for their [`NIID-Bench`](https://github.com/Xtra-Computing/NIID-Bench) repository, which has been instrumental in the development of our project. Our codebase is built upon the foundational work provided by their extensive research and resources in non-independent and identically distributed (non-i.i.d.) data for federated learning. We appreciate the opportunity to contribute to the ongoing dialogue in this field and thank the Xtra-Computing Group for their valuable contributions to the community.

## License

[(Back to top)](#table-of-contents)

[MIT license](./LICENSE_FNR-FL)

























































[Back to top](#table-of-contents)
