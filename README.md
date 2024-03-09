# $GRU^{spa}$: Gated Recurrent Unit with Spatial Attention for Spatio-Temporal Disaggregation

Open data is frequently released spatially aggregated, usually to comply with privacy policies.  But coarse, heterogeneous aggregations complicate learning and integration for downstream AI/ML systems.  In this work, we consider models to disaggregate spatio-temporal data from a low-resolution, irregular partition (e.g., census tract) to a high-resolution, irregular partition (e.g., city block). 
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/geo-boundaries.png)

We propose a model, Gated Recurrent Unit with Spatial Attention $(GRU^{spa})$, where spatial attention layers are integrated into the original Gated Recurrent Unit (GRU) model. The spatial attention layers capture spatial interactions among regions, while the gated recurrent module captures the temporal dependencies. Additionally, we utilize containment relationships between different geographic levels (e.g., when a given city block is wholly contained in a given census tract) to constrain the spatial attention layers. 
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/setting.png)

For situations where limited historical training data is available, we study transfer learning scenarios and show that a model pre-trained on one city variable can be fine-tuned for another city variable using only a few hundred samples. Evaluating these techniques on two mobility datasets, we find that $(GRU^{spa})$ provides a significant improvement over other neural models as well as typical heuristic methods, allowing us to synthesize realistic point data over small regions useful for training downstream models.
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/bikeshare_finetune_results.png)
