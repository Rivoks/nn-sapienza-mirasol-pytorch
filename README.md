# Mirasol3B model implementation

I provide in this project an implementation of the model Mirasol presented in the paper [MIRASOL3B: A MULTIMODAL AUTOREGRESSIVE MODEL FOR TIME-ALIGNED AND CONTEXTUAL MODALITIES](https://arxiv.org/pdf/2311.05698.pdf).

### Description of the method

The Mirasol method is an approach for integrating and processing multimodal data, such as audio, video and text, for comprehension and content generation tasks. The implementation focuses on transforming multimodal inputs into latent representations using specific encoders, then combining them via a combination module. Textual context processing is carried out using cross-attention to integrate contextual information. An autoregressive decoder is used to generate predictions based on the combined representations. The aim is to enable the machine to understand and generate content by capturing the nuances of different input modalities.

### How to install

In order to run the code it will be needed to install few packages:
```zsh
git clone https://github.com/Rivoks/nn-sapienza-mirasol-pytorch
pip3 install torch torchvision torchaudio av json matplotlib
```
