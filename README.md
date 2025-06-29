# <p align=center>`GeoAI in ICCV 2025`</p>

:star2:**A collection of papers related to Geo-spatial Information Science in ICCV 2025.**

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2025.06.29 :fire::fire::fire:
- **2025.06.29**: Update 1 papers.


## :memo: ICCV 2025 Accepted Paper List



#### SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images

> Gencer Sumbul, Chang Xu, Emanuele Dalsasso, Devis Tuia

* Paper: https://arxiv.org/abs/2506.19585
* Project: https://gsumbul.github.io/SMARTIES/
* <details>
    <summary>Abstract (Click to expand):</summary>
  From optical sensors to microwave radars, leveraging the complementary strengths of remote sensing (RS) sensors is crucial for achieving dense spatio-temporal monitoring of our planet. In contrast, recent deep learning models, whether task-specific or foundational, are often specific to single sensors or to fixed combinations: adapting such models to different sensory inputs requires both architectural changes and re-training, limiting scalability and generalization across multiple RS sensors. On the contrary, a single model able to modulate its feature representations to accept diverse sensors as input would pave the way to agile and flexible multi-sensor RS data processing. To address this, we introduce SMARTIES, a generic and versatile foundation model lifting sensor-specific/dependent efforts and enabling scalability and generalization to diverse RS sensors: SMARTIES projects data from heterogeneous sensors into a shared spectrum-aware space, enabling the use of arbitrary combinations of bands both for training and inference. To obtain sensor-agnostic representations, we train a single, unified transformer model reconstructing masked multi-sensor data with cross-sensor token mixup. On both single- and multi-modal tasks across diverse sensors, SMARTIES outperforms previous models that rely on sensor-specific pertaining. 
  </details>


#### When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning

> Junwei Luo, Yingying Zhang, Xue Yang, Kang Wu, Qi Zhu, Lei Liang, Jingdong Chen, Yansheng Li

* Paper: https://arxiv.org/abs/2503.07588
* Code: https://github.com/VisionXLab/LRS-VQA
* Dataset: https://huggingface.co/datasets/ll-13/LRS-VQA
* <details>
    <summary>Abstract (Click to expand):</summary>
    Efficient vision-language understanding of large Remote Sensing Images (RSIs) is meaningful but challenging. Current Large Vision-Language Models (LVLMs) typically employ limited pre-defined grids to process images, leading to information loss when handling gigapixel RSIs. Conversely, using unlimited grids significantly increases computational costs. To preserve image details while reducing computational complexity, we propose a text-guided token pruning method with Dynamic Image Pyramid (DIP) integration. Our method introduces: (i) a Region Focus Module (RFM) that leverages text-aware region localization capability to identify critical vision tokens, and (ii) a coarse-to-fine image tile selection and vision token pruning strategy based on DIP, which is guided by RFM outputs and avoids directly processing the entire large imagery. Additionally, existing benchmarks for evaluating LVLMs' perception ability on large RSI suffer from limited question diversity and constrained image sizes. We construct a new benchmark named LRS-VQA, which contains 7,333 QA pairs across 8 categories, with image length up to 27,328 pixels. Our method outperforms existing high-resolution strategies on four datasets using the same data. Moreover, compared to existing token reduction methods, our approach demonstrates higher efficiency under high-resolution settings.
  </details>


#### Dynamic Dictionary Learning for Remote Sensing Image Segmentation

> Xuechao Zou, Yue Li, Shun Zhang, Kai Li, Shiying Wang, Pin Tao, Junliang Xing, Congyan Lang

* Paper: https://arxiv.org/abs/2503.06683
* Project: https://xavierjiezou.github.io/D2LS/
* <details>
    <summary>Abstract (Click to expand):</summary>
    Remote sensing image segmentation faces persistent challenges in distinguishing morphologically similar categories and adapting to diverse scene variations. While existing methods rely on implicit representation learning paradigms, they often fail to dynamically adjust semantic embeddings according to contextual cues, leading to suboptimal performance in fine-grained scenarios such as cloud thickness differentiation. This work introduces a dynamic dictionary learning framework that explicitly models class ID embeddings through iterative refinement. The core contribution lies in a novel dictionary construction mechanism, where class-aware semantic embeddings are progressively updated via multi-stage alternating cross-attention querying between image features and dictionary embeddings. This process enables adaptive representation learning tailored to input-specific characteristics, effectively resolving ambiguities in intra-class heterogeneity and inter-class homogeneity. To further enhance discriminability, a contrastive constraint is applied to the dictionary space, ensuring compact intra-class distributions while maximizing inter-class separability. Extensive experiments across both coarse- and fine-grained datasets demonstrate consistent improvements over state-of-the-art methods, particularly in two online test benchmarks (LoveDA and UAVid).
  </details>












