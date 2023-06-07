# Urban Spatiotemporal Data Synthesis via Neural Disaggregation

The level of granularity of open data often conflicts the benefits it can provide. Less granular data can protect individual privacy, but to certain degrees, sabotage the promise of open data to promote transparency and assist research. Similar in the urban setting, aggregated urban data at high-level geographic units can mask out the underline particularities of city dynamics that may vary at lower areal levels. In this work, we aim to synthesize fine-grained, high resolution urban data, by breaking down aggregated urban data at coarse, low resolution geographic units. The goal is to increase the usability and realize the values as much as possible of highly aggregated urban data. 

![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/geo-visualizations.png)

- We leveraged neural methods to break down aggregated urban data, which outperformed traditional, simple disaggregation methods that are performed under simple assumptions. 

- We proposed a training approach for disaggregation task, **Chain-of-Training (COT)**, that can be incorporated into any of the training-based models. When we disaggregate from low to high geographic resolution, we add transitional disaggregation steps by incorporating intermediate geographic dimensions, and we backpropergate the loss at those transitional steps.

- We adapted the idea of restoration/reconstruction (**REC**) from super-resolution domain in our disaggregation case --- together with **COT** training procedure, we first disaggregate from low to high geographic level, and then re-aggregate back to the low level from our generated high level values. We calculate disaggregation losses at both high and intermediate levels, as well as reconstruction losses.

![alt text](https://github.com/BeanHam/2023-urban-disaggregation/blob/main/figures/reconstruction.png)
