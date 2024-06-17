# SARN: Structurally-Aware Recurrent Network for Spatio-Temporal Disaggregation

Open data is frequently released spatially aggregated, usually to comply with privacy policies.  But coarse, heterogeneous aggregations complicate learning and integration for downstream AI/ML systems.  In this work, we consider models to disaggregate spatio-temporal data from a low-resolution, irregular partition (e.g., census tract) to a high-resolution, irregular partition (e.g., city block). 
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/geo-boundaries.png)


We propose an overarching model named the Structurally-Aware Recurrent Network (SARN), which integrates structurally-aware spatial attention (SASA) layers into the Gated Recurrent Unit (GRU) model. The spatial attention layers capture spatial interactions among regions, while the gated recurrent module captures the temporal dependencies. Each SASA layer calculates both global and structural attention --- global attention facilitates comprehensive interactions between different geographic levels, while structural attention leverages the containment relationship between different geographic levels (e.g., a city block being wholly contained within a census tract) to ensure coherent and consistent results. 
![alt text](https://github.com/BeanHam/2024-urban-disaggregation/blob/main/figures/sarn.png)

For situations where limited historical training data is available, we study transfer learning scenarios and show that a model pre-trained on one city variable can be fine-tuned for another city variable using only a few hundred samples. Evaluating these techniques on two mobility datasets, we find that $(GRU^{spa})$ provides a significant improvement over other neural models as well as typical heuristic methods, allowing us to synthesize realistic point data over small regions useful for training downstream models.

For scenarios with limited historical training data, we explore transfer learning and show that a model pre-trained on one city variable can be fine-tuned for another city variable using only a few hundred samples.
![alt text](https://github.com/BeanHam/2024-urban-disaggregation/blob/main/figures/bikeshare-transfer-learning.png)
