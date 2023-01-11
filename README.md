# Car_Damage_Detection_Via_Unsupervised_Generative_Adversarial_Networks
INM363 Individual Project Report for MSc Data Science at the City, University of London (2021-2022)

This repository contains the work products INM363 Individual Project Report for MSc Data Science at the City, University of London (2021-2022).

Abstract: 

The purpose of this project was to solve for anomaly detection problem via an adversarial learning process using a real-world image dataset. The study was focused on the prospect of car damage detection based on the reconstruction of uncorrupted clean samples from any arbitrary image through an unsupervised Generative Adversarial Network (GAN) training process. This work studies car damage detection by researching over two baseline model architectures: GANomaly (Akcay et al., 2018) and SkipGANomaly (Akçay et al., 2019). The aim of this research was to test both models on a much more complex real-world image dataset and improve upon the results to add to the body of knowledge. The research was able to produce a variant of traditional GANomaly methods via the use of advanced training techniques referenced in the previous research literature. Most importantly, a new variant of the SkipGANomaly method was proposed by iteratively projecting arbitrary input towards the clean distribution in the target domain through the introduction of synthetic anomalies in the form of masked regions. By reformulating the original reconstruction task of the SkipGANomaly method as an image completion problem, we were able to produce unprecedented results in successfully completing the masked image regions with the anomaly-free versions. We were able to demonstrate the relevance of our approach with a consistent assessment of the reconstruction-based image completion method, by comparing its performance over a complex client dataset. This study provides substantial proof of concept demonstrating the potential GAN frameworks have to offer towards unsupervised anomaly detection in complex and multifaceted image datasets. 

Keywords: Generative Adversarial Learning, Anomaly Detection, GANomaly, SkipGANomaly, Image Reconstruction, Deep Neural Networks

The dataset used in this project is a proprietary dataset that Tractable AI acquired in 2019 from a third-party software development company. The Nugen dataset contains ~9 million .jpeg images obtained from historic auto collision claim reports. The images are categorised by claim number, each containing multiple photographs of the same vehicle from different vantage points (usually taken by the repair shop or the owner). Each image is given a one-hot-encoded label (1 for True, 0 for False), based on the parts present in each image and whether or not they are damaged. The original raw data is stored in an S3 Object Storage solution offered by Amazon Web Services (AWS). Nugen_dataset folder contains the final list of filenames used in the training (undamaged) and testing (damaged). The csv files of the project matadate of the Nugen dataset is also provided in the Nugen_dataset folder. 


The notebooks are based on the original GANomaly architecture proposed by Akcay et al, in 2018 (https://arxiv.org/abs/1805.06725). The model employs DCGAN architecutre (https://arxiv.org/abs/1511.06434) to create a new anomaly detection model using encoder-decoder-encoder sub-networks that generates high-dimensional image space. The model is trained solely on normal data (in our case undamaged car images), and it attemps to map the input image to a representative latent space, which is then used to reconstruct the generated output image. To map the generated image back to its latent representation, additional encoder network is used. The distance between the input image and the generated image, as well as their latent representations, are minimised during training.

<img src="model_figures_png/GANomaly.png" width="800" height="400">
- <i>Figure A. GANomaly model architechture with Encoder Decoder and Encoder network and the discriminator. </i>
    
The original architecture was built using a PyTorch library which is publicly available in a GitHub repository (https://github.com/samet-akcay/ganomaly). To gain a deeper familiarity with the model and more leeway for customization, we programmed the model architecture from scratch using TensorFlow and Keras libraries.

This notebook uses an update version of the GANomaly proposed by Akcay et al, in 2019 (https://arxiv.org/abs/1901.08954), where skip connections are added between the encoder and the decoder of the Generator network. Additionally, the second encoder network is removed from the model. Furthermore a different loss function is applied for backpropogation. 

<img src="model_figures_png/SkipGANomaly.png" width="800" height="400">
<img src="model_figures_png/SkipGANomaly2.png" width="800" height="400">
- <i>Figure B. SkipGANomaly model architechture with Encoder-Decoder Generator network and the discriminator. </i>

The project consists of 5 experiments, each experiment grouped in different folders (Method_1 to Method_5). 
<ol>
    <li>Method 1: contains the implementation of the original GANomaly baseline model</li>
    <li>Method 2: contains a modified implementation of the original GANomaly model with the improvements indicated below</li>

        <ul>
            <li> larger latent dimension (512)</li>
            <li> larger kernel size (5x5)</li>
            <li> soft and noisy labels</li>
            <li> modified adversarial loss function g_adv</li>
            <li> adding noise to the discriminator</li>
            <li> Two Time-scale Update Rule (TTUR)</li>
        </ul>
    
    <li>Method 3: contains the improved version of the modified GANomaly framework (method 2) with the addition of Spectral Normalisation to the Discriminator network's convolutional kernel</li>
    <li>Method 4: contains the implementation of the original SkipGANomaly baseline model</li>
    <li>Method 5: contains the masked implementation of the original SkipGANomaly method with the introduction of random 128x128 masked patches in the arbitrary input image. Figure below shows the simple pipeline of the proposed method. </li>
</ol>

<img width="800" alt="image" src="https://user-images.githubusercontent.com/27391785/211730896-ebdabfd5-7c72-4908-b977-e7077ac3cea1.png">

