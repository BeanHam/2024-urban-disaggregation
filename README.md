# GRU^{spa}: Gated Recurrent Unit with Spatial Attention for Spatio-Temporal Disaggregation

Open data is frequently released spatially and temporally aggregated, usually to comply with privacy policies.  Varying aggregation levels (e.g., zip code, census tract, city block) complicate the integration across variables needed to provide multi-variate training sets for downstream AI/ML systems. In this work, we consider models to disaggregate spatial data, learning a function from a low-resolution irregular partition (e.g., zip code) to s high-resolution irregular partition (e.g., city block).  

![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/geo-boundaries.png)

We propose a hierarchical architecture that aligns each geographic aggregation level with a layer in the network such that all aggregation levels can be learned simultaneously by including loss terms for all intermediate levels as well as the final output.  We then consider additional loss terms that compare the re-aggregated output against ground truth to further improve performance. To balance the tradeoff between training time and accuracy, we consider three training regimes, including a layer-by-layer process that achieves competitive predictions with significantly reduced training time. 

![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/setting.png)

For situations where limited historical training data is available, we study transfer learning scenarios and show that a model pre-trained on one city variable can be fine-tuned for another city variable using only a few hundred samples, highlighting the common dynamics among variables from the same built environment and underlying population. Evaluating these techniques on four datasets across two cities, three variables, and two application domains, we find that geographically coherent architectures provide a significant improvement over baseline models as well as typical heuristic methods, advancing our long-term goal of synthesizing any variable, at any location, at any resolution.

![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/bikeshare_finetune_results.png)
![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/911-call_finetune_results.png)
