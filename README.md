# <p align=center>`GeoAI in ICCV 2025`</p>

:star2:**A collection of papers related to Geo-spatial Information Science in ICCV 2025.**

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2025.06.29 :fire::fire::fire:
- **2025.06.29**: Update 9 papers.


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



#### Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling

> Fengxiang Wang, Hongzhen Wang, Di Wang, Zonghao Guo, Zhenyu Zhong, Long Lan, Wenjing Yang, Jing Zhang

* Paper: https://arxiv.org/abs/2406.11933
* Code: https://github.com/MiliLab/SelectiveMAE
* <details>
    <summary>Abstract (Click to expand):</summary>
    Masked Image Modeling (MIM) has become an essential method for building foundational visual models in remote sensing (RS). However, the limitations in size and diversity of existing RS datasets restrict the ability of MIM methods to learn generalizable representations. Additionally, conventional MIM techniques, which require reconstructing all tokens, introduce unnecessary computational overhead. To address these issues, we present a new pre-training pipeline for RS models, featuring the creation of a large-scale RS dataset and an efficient MIM approach. We curated a high-quality dataset named \textbf{OpticalRS-13M} by collecting publicly available RS datasets and processing them through exclusion, slicing, and deduplication. OpticalRS-13M comprises 13 million optical images covering various RS tasks, such as object detection and pixel segmentation. To enhance efficiency, we propose \textbf{SelectiveMAE}, a pre-training method that dynamically encodes and reconstructs semantically rich patch tokens, thereby reducing the inefficiencies of traditional MIM models caused by redundant background pixels in RS images. Extensive experiments show that OpticalRS-13M significantly improves classification, detection, and segmentation performance, while SelectiveMAE increases training efficiency over 2 times. This highlights the effectiveness and scalability of our pipeline in developing RS foundational models.
  </details>



#### RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model

> Huiyang Hu, Peijin Wang, Hanbo Bi, Boyuan Tong, Zhaozhi Wang, Wenhui Diao, Hao Chang, Yingchao Feng, Ziqi Zhang, Yaowei Wang, Qixiang Ye, Kun Fu, Xian Sun

* Paper: https://arxiv.org/abs/2411.17984
* <details>
    <summary>Abstract (Click to expand):</summary>
    Remote sensing foundation models largely break away from the traditional paradigm of designing task-specific models, offering greater scalability across multiple tasks. However, they face challenges such as low computational efficiency and limited interpretability, especially when dealing with large-scale remote sensing images. To overcome these, we draw inspiration from heat conduction, a physical process modeling local heat diffusion. Building on this idea, we are the first to explore the potential of using the parallel computing model of heat conduction to simulate the local region correlations in high-resolution remote sensing images, and introduce RS-vHeat, an efficient multi-modal remote sensing foundation model. Specifically, RS-vHeat 1) applies the Heat Conduction Operator (HCO) with a complexity of  and a global receptive field, reducing computational overhead while capturing remote sensing object structure information to guide heat diffusion; 2) learns the frequency distribution representations of various scenes through a self-supervised strategy based on frequency domain hierarchical masking and multi-domain reconstruction; 3) significantly improves efficiency and performance over state-of-the-art techniques across 4 tasks and 10 datasets. Compared to attention-based remote sensing foundation models, we reduce memory usage by 84\%, FLOPs by 24\% and improves throughput by 2.7 times. The code will be made publicly available.
  </details>


#### SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World

> Chen Chen, Zhirui Wang, Taowei Sheng, Yi Jiang, Yundu Li, Peirui Cheng, Luning Zhang, Kaiqiang Chen, Yanfeng Hu, Xue Yang, Xian Sun

* Paper: https://arxiv.org/abs/2503.16399
* Code: https://github.com/chenchen235/SA-Occ
* <details>
    <summary>Abstract (Click to expand):</summary>
    Existing vision-based 3D occupancy prediction methods are inherently limited in accuracy due to their exclusive reliance on street-view imagery, neglecting the potential benefits of incorporating satellite views. We propose SA-Occ, the first Satellite-Assisted 3D occupancy prediction model, which leverages GPS & IMU to integrate historical yet readily available satellite imagery into real-time applications, effectively mitigating limitations of ego-vehicle perceptions, involving occlusions and degraded performance in distant regions. To address the core challenges of cross-view perception, we propose: 1) Dynamic-Decoupling Fusion, which resolves inconsistencies in dynamic regions caused by the temporal asynchrony between satellite and street views; 2) 3D-Proj Guidance, a module that enhances 3D feature extraction from inherently 2D satellite imagery; and 3) Uniform Sampling Alignment, which aligns the sampling density between street and satellite views. Evaluated on Occ3D-nuScenes, SA-Occ achieves state-of-the-art performance, especially among single-frame methods, with a 39.05% mIoU (a 6.97% improvement), while incurring only 6.93 ms of additional latency per frame.
  </details>


#### HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery

> Yu Wang, Bo Dang, Wanchun Li, Wei Chen, Yansheng Li

* Paper: https://image-1305984033.cos.ap-nanjing.myqcloud.com/pdf/2025_iccv_vector.pdf
* Code: https://github.com/vvangfaye/HoliTracer
* <details>
    <summary>Abstract (Click to expand):</summary>
    With the increasing resolution of remote sensing imagery(RSI), large-size RSI has emerged as a vital data source for high-precision vector mapping of geographic objects. Existing methods are typically constrained to processing small image patches, which often leads to the loss of contextual information and produces fragmented vector outputs. To address these, this paper introduces HoliTracer, the first framework designed to holistically extract vectorized geographic objects from large-size RSI. In HoliTracer, we enhance segmentation of large-size RSI using the Context Attention Net (CAN), which employs a local-to-global attention mechanism to capture contextual dependencies. Furthermore, we achieve holistic vectorization through a robust pipeline that leverages the Mask Contour Reformer(MCR) to reconstruct polygons and the Polygon Sequence Tracer (PST) to trace vertices. Extensive experiments on large-size RSI datasets, including buildings, water bodies, and roads, demonstrate that HoliTracer outperforms stateof-the-art methods. 
  </details>


#### Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis

> Junyan Ye, Jun He, Weijia Li, Zhutao Lv, Yi Lin, Jinhua Yu, Haote Yang, Conghui He

* Paper: https://arxiv.org/abs/2408.01812
* Code: https://github.com/opendatalab/skydiffusion
* <details>
    <summary>Abstract (Click to expand):</summary>
    Ground-to-aerial image synthesis focuses on generating realistic aerial images from corresponding ground street view images while maintaining consistent content layout, simulating a top-down view. The significant viewpoint difference leads to domain gaps between views, and dense urban scenes limit the visible range of street views, making this cross-view generation task particularly challenging. In this paper, we introduce SkyDiffusion, a novel cross-view generation method for synthesizing aerial images from street view images, utilizing a diffusion model and the Bird's-Eye View (BEV) paradigm. The Curved-BEV method in SkyDiffusion converts street-view images into a BEV perspective, effectively bridging the domain gap, and employs a "multi-to-one" mapping strategy to address occlusion issues in dense urban scenes. Next, SkyDiffusion designed a BEV-guided diffusion model to generate content-consistent and realistic aerial images. Additionally, we introduce a novel dataset, Ground2Aerial-3, designed for diverse ground-to-aerial image synthesis applications, including disaster scene aerial synthesis, low-altitude UAV image synthesis, and historical high-resolution satellite image synthesis tasks. Experimental results demonstrate that SkyDiffusion outperforms state-of-the-art methods on cross-view datasets across natural (CVUSA), suburban (CVACT), urban (VIGOR-Chicago), and various application scenarios (G2A-3), achieving realistic and content-consistent aerial image generation.
  </details>



#### Towards Privacy-preserved Pre-training of Remote Sensing Foundation Models with Federated Mutual-guidance Learning

> Jieyi Tan, Chengwei Zhang, Bo Dang, Yansheng Li

* Paper: https://arxiv.org/abs/2503.11051
* <details>
    <summary>Abstract (Click to expand):</summary>
    Traditional Remote Sensing Foundation Models (RSFMs) are pre-trained with a data-centralized paradigm, through self-supervision on large-scale curated remote sensing data. For each institution, however, pre-training RSFMs with limited data in a standalone manner may lead to suboptimal performance, while aggregating remote sensing data from multiple institutions for centralized pre-training raises privacy concerns. Seeking for collaboration is a promising solution to resolve this dilemma, where multiple institutions can collaboratively train RSFMs without sharing private data. In this paper, we propose a novel privacy-preserved pre-training framework (FedSense), which enables multiple institutions to collaboratively train RSFMs without sharing private data. However, it is a non-trivial task hindered by a vicious cycle, which results from model drift by remote sensing data heterogeneity and high communication overhead. To break this vicious cycle, we introduce Federated Mutual-guidance Learning. Specifically, we propose a Server-to-Clients Guidance (SCG) mechanism to guide clients updates towards global-flatness optimal solutions. Additionally, we propose a Clients-to-Server Guidance (CSG) mechanism to inject local knowledge into the server by low-bit communication. Extensive experiments on four downstream tasks demonstrate the effectiveness of our FedSense in both full-precision and communication-reduced scenarios, showcasing remarkable communication efficiency and performance gains.
  </details>


#### TopicGeo: An Efficient Unified Framework for Geolocation

> TopicGeo: An Efficient Unified Framework for Geolocation

* Paper: None
* <details>
    <summary>Abstract (Click to expand):</summary>
    在小尺度的查询图像与大量大尺度的地理参考图像之间建立空间对应关系的视觉地理定位技术已受到广泛关注。现有方法通常采用“先检索再匹配”的分离范式，但该范式存在计算效率低或精度受限的问题。为此，我们提出了一个统一的检索匹配框架TopicGeo，通过三项关键创新实现查询图像与参考图像的直接且精确匹配。首先我们将通过CLIP提示学习和语义蒸馏提取的文本对象语义（称为Topic即主题）嵌入地理定位框架，以消除多时相遥感图像中类内与类间的分布差异，同时提升处理效率。然后基于中心自适应标签分配与离群点剔除机制作为联合“检索-匹配”优化策略，确保了任务一致的特征学习与精确的空间对应关系。我们还引入了多层次的精细匹配流程，以进一步提升匹配的质量和数量。在大规模的合成与真实数据集上的评估表明，TopicGeo在检索召回率和匹配精度方面均具有较好的性能，同时保持了良好的计算效率。
  </details>





