# ODT-AI-Filters
AI plug-ins for the Oligo Designer Toolsuite package


- oligo-designer-toolsuite
- pytorch
- pandas
- numoy
- matplotlib


For generating the dataset of the hybridization-probability filters the Nupack pakage is required (TODO: add instructions.)


The model training pipeline performs a grid hyperparameters seach and stores all the models trained in a folder [filter_type]/[model_architecture]/[dataset]. Then the best model is saved inthe same folder under the filter_type name.


The architecture used for the different ai filters are:

| AI filter | Architecture |
|-|-|
| hybridization probability | lstm |