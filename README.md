# SARN: Structurally-Aware Recurrent Network for Spatio-Temporal Disaggregation

### Introduction
Open data is frequently released spatially aggregated, usually to comply with privacy policies.  But coarse, heterogeneous aggregations complicate learning and integration for downstream AI/ML systems.  In this work, we consider models to disaggregate spatio-temporal data from a low-resolution, irregular partition (e.g., census tract) to a high-resolution, irregular partition (e.g., city block). 
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/geo-boundaries.png)

### Model Architecture
We propose an overarching model named the Structurally-Aware Recurrent Network (SARN), which integrates structurally-aware spatial attention (SASA) layers into the Gated Recurrent Unit (GRU) model. The spatial attention layers capture spatial interactions among regions, while the gated recurrent module captures the temporal dependencies. Each SASA layer calculates both global and structural attention --- global attention facilitates comprehensive interactions between different geographic levels, while structural attention leverages the containment relationship between different geographic levels (e.g., a city block being wholly contained within a census tract) to ensure coherent and consistent results. 
![alt text](https://github.com/BeanHam/2024-urban-disaggregation/blob/main/figures/sarn.png)

### Transfer Learning
For scenarios with limited historical training data, we explore transfer learning and show that a model pre-trained on one city variable can be fine-tuned for another city variable using only a few hundred samples. Evaluating these techniques on two mobility datasets, we find that on both datasets, SARN significantly outperforms other neural models (5\% and 1\%) and typical heuristic methods (40\% and 14\%), enabling us to generate realistic, high-quality fine-grained data for downstream applications.
![alt text](https://github.com/BeanHam/2024-urban-disaggregation/blob/main/figures/bikeshare-transfer-learning.png)
