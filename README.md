<img src="./Text2Video-Zero.png" width="850" alt="Architecture diagram of Text2Video-Zero" title="Architecture diagram of Text2Video-Zero"/>

# Text2Video-Zero
Implementation of [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/pdf/2303.13439.pdf)

```shell
pip install diffusers==0.14.0 transformers==4.26.0
# ControlNet
pip install git+https://github.com/patrickvonplaten/controlnet_aux.git
python generate.py
```

<img src="./images/panda.png" height="321" alt="Chinese Panda" title="Chinese Panda"/><img src="./images/pandas.png" height="321" alt="Chinese Panda" title="Chinese Panda"/>

**Version 1**  
<img src="./images/frames.png" width="1000" alt="Chinese Panda" title="Chinese Panda"/>

**Version 2** - Motion in Latents, No Cross-Frame Attention  
<img src="./images/frames-v2-1.png" width="1000" alt="Chinese Panda"/>
<img src="./images/frames-v2-2.png" width="1000" alt="a high quality realistic photo of a panda playing guitar on times square"/>

**Version 3** - Motion in Latents, Cross-Frame Attention  
<img src="./images/frames-v3-1.png" width="1000" alt="Chinese Panda"/>
<img src="./images/frames-v3-2.png" width="1000" alt="a high quality realistic photo of a panda playing guitar on times square"/>
<img src="./images/frames-v3-3.png" width="1000" alt="an astronaut is skiing down a hill"/>
