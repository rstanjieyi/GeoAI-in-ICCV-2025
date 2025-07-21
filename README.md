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


#### Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?

> Yuru Jia, Valerio Marsocci, Ziyang Gong, Xue Yang, Maarten Vergauwen, Andrea Nascetti

* Paper: https://arxiv.org/abs/2503.07890
* Code: https://github.com/yurujaja/SatDiFuser
* <details>
    <summary>Abstract (Click to expand):</summary>
    Self-supervised learning (SSL) has revolutionized representation learning in Remote Sensing (RS), advancing Geospatial Foundation Models (GFMs) to leverage vast unlabeled satellite imagery for diverse downstream tasks. Currently, GFMs primarily focus on discriminative objectives, such as contrastive learning or masked image modeling, owing to their proven success in learning transferable representations. However, generative diffusion models--which demonstrate the potential to capture multi-grained semantics essential for RS tasks during image generation--remain underexplored for discriminative applications. This prompts the question: can generative diffusion models also excel and serve as GFMs with sufficient discriminative power? In this work, we answer this question with SatDiFuser, a framework that transforms a diffusion-based generative geospatial foundation model into a powerful pretraining tool for discriminative RS. By systematically analyzing multi-stage, noise-dependent diffusion features, we develop three fusion strategies to effectively leverage these diverse representations. Extensive experiments on remote sensing benchmarks show that SatDiFuser outperforms state-of-the-art GFMs, achieving gains of up to +5.7% mIoU in semantic segmentation and +7.9% F1-score in classification, demonstrating the capacity of diffusion-based generative foundation models to rival or exceed discriminative GFMs. Code will be released.
  </details>




#### TerraMind: Large-Scale Generative Multimodality for Earth Observation

> Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, Rahul Ramachandran, Paolo Fraccaro, Thomas Brunschwiler, Gabriele Cavallaro, Juan Bernabe-Moreno, Nicolas Longépé

* Paper: https://arxiv.org/abs/2504.11171
* Project: https://ibm.github.io/terramind/
* <details>
    <summary>Abstract (Click to expand):</summary>
    We present TerraMind, the first any-to-any generative, multimodal foundation model for Earth observation (EO). Unlike other multimodal models, TerraMind is pretrained on dual-scale representations combining both token-level and pixel-level data across modalities. On a token level, TerraMind encodes high-level contextual information to learn cross-modal relationships, while on a pixel level, TerraMind leverages fine-grained representations to capture critical spatial nuances. We pretrained TerraMind on nine geospatial modalities of a global, large-scale dataset. In this paper, we demonstrate that (i) TerraMind's dual-scale early fusion approach unlocks a range of zero-shot and few-shot applications for Earth observation, (ii) TerraMind introduces "Thinking-in-Modalities" (TiM) -- the capability of generating additional artificial data during finetuning and inference to improve the model output -- and (iii) TerraMind achieves beyond state-of-the-art performance in community-standard benchmarks for EO like PANGAEA. The pretraining dataset, the model weights, and our code are open-sourced under a permissive license.
  </details>

#### GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks

> Muhammad Sohail Danish, Muhammad Akhtar Munir, Syed Roshaan Ali Shah, Kartik Kuckreja, Fahad Shahbaz Khan, Paolo Fraccaro, Alexandre Lacoste, Salman Khan

* Paper: https://arxiv.org/abs/2411.19325v2
* Project: https://the-ai-alliance.github.io/GEO-Bench-VLM/
* <details>
    <summary>Abstract (Click to expand):</summary>
    While numerous recent benchmarks focus on evaluating generic Vision-Language Models (VLMs), they do not effectively address the specific challenges of geospatial applications. Generic VLM benchmarks are not designed to handle the complexities of geospatial data, an essential component for applications such as environmental monitoring, urban planning, and disaster management. Key challenges in the geospatial domain include temporal change detection, large-scale object counting, tiny object detection, and understanding relationships between entities in remote sensing imagery. To bridge this gap, we present GEOBench-VLM, a comprehensive benchmark specifically designed to evaluate VLMs on geospatial tasks, including scene understanding, object counting, localization, fine-grained categorization, segmentation, and temporal analysis. Our benchmark features over 10,000 manually verified instructions and spanning diverse visual conditions, object types, and scales. We evaluate several state-of-the-art VLMs to assess performance on geospatial-specific challenges. The results indicate that although existing VLMs demonstrate potential, they face challenges when dealing with geospatial-specific tasks, highlighting the room for further improvements. Notably, the best-performing LLaVa-OneVision achieves only 41.7% accuracy on MCQs, slightly more than GPT-4o, which is approximately double the random guess performance.
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

#### Cross-modal Ship Re-Identification via Optical and SAR Imagery: A Novel Dataset and Method

> Han Wang, Shengyang Li, Jian Yang, Yuxuan Liu, Yixuan Lv, Zhuang Zhou

* Paper: https://arxiv.org/abs/2506.22027
* Code: https://github.com/Alioth2000/Hoss-ReID
* <details>
    <summary>Abstract (Click to expand):</summary>
    Detecting and tracking ground objects using earth observation imagery remains a significant challenge in the field of remote sensing. Continuous maritime ship tracking is crucial for applications such as maritime search and rescue, law enforcement, and shipping analysis. However, most current ship tracking methods rely on geostationary satellites or video satellites. The former offer low resolution and are susceptible to weather conditions, while the latter have short filming durations and limited coverage areas, making them less suitable for the real-world requirements of ship tracking. To address these limitations, we present the Hybrid Optical and Synthetic Aperture Radar (SAR) Ship Re-Identification Dataset (HOSS ReID dataset), designed to evaluate the effectiveness of ship tracking using low-Earth orbit constellations of optical and SAR sensors. This approach ensures shorter re-imaging cycles and enables all-weather tracking. HOSS ReID dataset includes images of the same ship captured over extended periods under diverse conditions, using different satellites of different modalities at varying times and angles. Furthermore, we propose a baseline method for cross-modal ship re-identification, TransOSS, which is built on the Vision Transformer architecture. It refines the patch embedding structure to better accommodate cross-modal tasks, incorporates additional embeddings to introduce more reference information, and employs contrastive learning to pre-train on large-scale optical-SAR image pairs, ensuring the model's ability to extract modality-invariant features. 
  </details>



#### Where am I? Cross-View Geo-localization with Natural Language Descriptions

> Junyan Ye, Honglin Lin, Leyan Ou, Dairong Chen, Zihao Wang, Conghui He, Weijia Li

* Paper: https://arxiv.org/abs/2412.17007v1
* Project: https://yejy53.github.io/CVG-Text/
* <details>
    <summary>Abstract (Click to expand):</summary>
    Cross-view geo-localization identifies the locations of street-view images by matching them with geo-tagged satellite images or OSM. However, most studies focus on image-to-image retrieval, with fewer addressing text-guided retrieval, a task vital for applications like pedestrian navigation and emergency response. In this work, we introduce a novel task for cross-view geo-localization with natural language descriptions, which aims to retrieve corresponding satellite images or OSM database based on scene text. To support this task, we construct the CVG-Text dataset by collecting cross-view data from multiple cities and employing a scene text generation approach that leverages the annotation capabilities of Large Multimodal Models to produce high-quality scene text descriptions with localization this http URL, we propose a novel text-based retrieval localization method, CrossText2Loc, which improves recall by 10% and demonstrates excellent long-text retrieval capabilities. In terms of explainability, it not only provides similarity scores but also offers retrieval reasons. 
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



#### Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image

> Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng

* Paper: https://arxiv.org/abs/2412.04201
* <details>
    <summary>Abstract (Click to expand):</summary>
    Hyperspectral images (HSIs) are frequently noisy and of low resolution due to the constraints of imaging devices. Recently launched satellites can concurrently acquire HSIs and panchromatic (PAN) images, enabling the restoration of HSIs to generate clean and high-resolution imagery through fusing PAN images for denoising and super-resolution. However, previous studies treated these two tasks as independent processes, resulting in accumulated errors. This paper introduces \textbf{H}yperspectral \textbf{I}mage Joint \textbf{Pand}enoising \textbf{a}nd Pan\textbf{s}harpening (Hipandas), a novel learning paradigm that reconstructs HRHS images from noisy low-resolution HSIs (LRHS) and high-resolution PAN images. The proposed zero-shot Hipandas framework consists of a guided denoising network, a guided super-resolution network, and a PAN reconstruction network, utilizing an HSI low-rank prior and a newly introduced detail-oriented low-rank prior. The interconnection of these networks complicates the training process, necessitating a two-stage training strategy to ensure effective training. Experimental results on both simulated and real-world datasets indicate that the proposed method surpasses state-of-the-art algorithms, yielding more accurate and visually pleasing HRHS images.
  </details>


#### UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding

> Jie Feng, Shengyuan Wang, Tianhui Liu, Yanxin Xi, Yong Li

* Paper: https://arxiv.org/abs/2506.23219
* Code: https://github.com/tsinghua-fib-lab/UrbanLLaVA
* <details>
    <summary>Abstract (Click to expand):</summary>
    Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce , a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In , we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of  across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that  outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. 
  </details>

#### TopicGeo: An Efficient Unified Framework for Geolocation

> TopicGeo: An Efficient Unified Framework for Geolocation

* Paper: None
* <details>
    <summary>Abstract (Click to expand):</summary>
    在小尺度的查询图像与大量大尺度的地理参考图像之间建立空间对应关系的视觉地理定位技术已受到广泛关注。现有方法通常采用“先检索再匹配”的分离范式，但该范式存在计算效率低或精度受限的问题。为此，我们提出了一个统一的检索匹配框架TopicGeo，通过三项关键创新实现查询图像与参考图像的直接且精确匹配。首先我们将通过CLIP提示学习和语义蒸馏提取的文本对象语义（称为Topic即主题）嵌入地理定位框架，以消除多时相遥感图像中类内与类间的分布差异，同时提升处理效率。然后基于中心自适应标签分配与离群点剔除机制作为联合“检索-匹配”优化策略，确保了任务一致的特征学习与精确的空间对应关系。我们还引入了多层次的精细匹配流程，以进一步提升匹配的质量和数量。在大规模的合成与真实数据集上的评估表明，TopicGeo在检索召回率和匹配精度方面均具有较好的性能，同时保持了良好的计算效率。
  </details>





