# A Comprehensive Benchmark for Defenses Against Black-Box Adversarial Attacks

This repository contains the code for the study carried out in my bachelor thesis: LINK

The study explores several relevant defense approaches against adversarial attacks and evaluates all of them in a unified
environment using an iterative decision-based attack, an improved version of Boundary
Attack, the HopSkipJumpAttack. It is model-agnostic and realistic due to the limited information the attacker needs. Experiments are carried out on the datasets MNIST and Cifar10.

## Used Attacks and Evaluated Defenses
- HopSkipJumpAttack:
    - Paper: https://arxiv.org/abs/1904.02144
    - Used Implementation: https://foolbox.readthedocs.io/en/latest/modules/attacks/decision.html#foolbox.attacks.BoundaryAttackPlusPlus
<br/>

- Defense-GAN:
    - Paper: https://arxiv.org/abs/1805.06605
    - Implementation: https://github.com/kabkabm/defensegan
<br/>

- Ensemble Adversarial Training:
    - Paper: https://arxiv.org/abs/1705.07204
    - Implementation: https://github.com/ftramer/ensemble-adv-training
<br/>

- Mahalanobis Detector:
    - Paper: https://arxiv.org/abs/1807.03888
    - Implementation: https://github.com/pokaxpoka/deep_Mahalanobis_detector
<br/>

- RCE + K-Density Detector:
    - Paper: https://arxiv.org/abs/1706.00633
    - Implementation: https://github.com/P2333/Reverse-Cross-Entropy and https://github.com/rfeinman/detecting-adversarial-samples

## Prerequisites
The code is tested in the following enviroment:

- Python 3.6
- GPU with Cuda 9 installed
- All necessary python requirements can be installed using the requirements.txt file in the root of the repository.

## Result Generation
For regenerating the results presented in the thesis the file evaluate.py in the root of the repository has to be run with the according configuration file. The configuration files can be found in the config_files/ subfolder and are named after the defense mechanism and the dataset. E.g. run the following command:
    
    python evaluate.py config_files/mnist_EnsAdvTraining.yaml