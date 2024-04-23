# ODT-AI-Filters
This repository is an AI plug-ins for the Oligo Designer Toolsuite package. Here we collect all the required functionalities and implementation to train and run machine learning models in the Oligo Designer Toolsuite pipelines.

For each task we provide a pretrained model, but also the code impelemtation to train you own model with the architecture and hypeparameters you prefer.
In general, the model training pipeline performs a grid hyperparameters seach and stores all the models trained in a folder [filter_type]/[model_architecture]/[dataset_name]. Then the best model is saved inthe same folder under the filter_type name.


## Available AI models.


### Hybridization Probability

This model is used to improove the specificity estiamtion of the oligo sequences.
With a Recurrent Neural Network we estimate the hybridization probability between off-target sites and oligos, and use it to determine if the sites represent a real threat. To predict hybridization probability, our models use the genomic sequences of the oligo and the off-target site. In addition, several manually extracted sequence features, such as the GC content and the melting temperature, were fed into the model.

For generating the ground-truths of the hybridization-probability filters we use the [NUPACK](https://docs.nupack.org/) pakage, which estimasates the equilibrium cooncentrations of DNA complexes.
In particualr, the score is obtained from the final concentration of DNA complexes in NUPACK tube experiment simulation
that contains the oligo sequence, the exact on-target region and the off-target. The oligo, on-target and off-target
strands are initially set at the same concentration $C_{in}$ and we define the duplexing score as: 

$log( \dfrac{C_{oligo + off-t}}{C_{oligo + off-t}  + C_{oligo + on-t} })$. 
    
## Pretrained models

The best performing architecture for the different taks that are available as pretrained models are the following:

| AI filter | Architecture |
|-|-|
| hybridization probability | lstm |