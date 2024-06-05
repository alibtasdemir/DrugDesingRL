<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">De Novo Drug Design with Deep Q-Learning</h3>
<a name="readme-top"></a>
  <p align="center">
    ----
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The discovery of new drugs is a complex, costly, and time-consuming process. Traditional methods often fall short in efficiently navigating the vast chemical space, estimated to contain around $10^{60}$ possible small molecules. In this paper, we propose a novel approach to de novo drug design using deep reinforcement learning. Our model employs a deep Q-learning framework to iteratively generate and optimize drug- like molecules, guaranteeing 100% chemical validity by defining specific action boundaries within the molecular modification process. Our method represents the drug discovery process as a Markov Decision Process (MDP) and utilizes a double deep Q-network (DQN) to predict the optimal molecular modifications. We validate our approach through extensive experiments and compare its performance with several state-of-the-art models, demonstrating that our method can effectively generate high-quality, valid molecules with desirable drug-like properties. Although our approach does not yet surpass existing methods in terms of Quantitative Estimate of Drug-likeness (QED) scores, it highlights the potential of reinforcement learning in automating and enhancing the drug discovery pipeline. Future work will focus on refining the reward functions and incorporating additional molecular features to further improve the efficacy of the generated compounds.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[![Pytorch][Pytorch]][Pytorch-url] [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To test the framework on local follow these steps.

### Prerequisites & Installation

The requirements provided in `reqiurements.txt`. 

1. Clone the repo
   ```bash
   git clone https://github.com/alibtasdemir/DrugDesignRL.git
   cd DrugDesignRL
   ```
2. Create a virtual environment (conda)
   ```bash
   conda create --name <env> --file requirements.txt
   conda activate <env>
   ```
3. To train the network run `train.py` (Check arguments to modify training)
   ```bash
   python train.py --exp_name [EXPERIMENT_NAME] --allow_removal --allow_no_modification
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Inference

1. To run the model in inference mode run `inference.py` (Check arguments to modify training)
   ```bash
   python inference.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Ali Baran Taşdemir - [@alibrnt](https://twitter.com/alibrnt) - alibaran@tasdemir.us

Project Link: [https://github.com/alibtasdemir/DrugDesignRL](https://github.com/alibtasdemir/DrugDesignRL)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/alibtasdemir/DrugDesignRL.svg?style=for-the-badge
[contributors-url]: https://github.com/alibtasdemir/DrugDesignRL/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alibtasdemir/DrugDesignRL.svg?style=for-the-badge
[forks-url]: https://github.com/alibtasdemir/DrugDesignRL/network/members
[stars-shield]: https://img.shields.io/github/stars/alibtasdemir/DrugDesignRL.svg?style=for-the-badge
[stars-url]: https://github.com/alibtasdemir/DrugDesignRL/stargazers
[issues-shield]: https://img.shields.io/github/issues/alibtasdemir/DrugDesignRL.svg?style=for-the-badge
[issues-url]: https://github.com/alibtasdemir/DrugDesignRL/issues
[license-shield]: https://img.shields.io/github/license/alibtasdemir/DrugDesignRL.svg?style=for-the-badge
[license-url]: https://github.com/alibtasdemir/DrugDesignRL/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alibtasdemir
[product-screenshot]: images/screenshot.png
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

