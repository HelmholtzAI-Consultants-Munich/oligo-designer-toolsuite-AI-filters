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

## Scores definition for each filter.

### hybridization probability
The duplexing score is obtained from the final concentration of DNA complexes in NUPACK tube experiment simulation
that contains the oligo sequence, the exact on-target region and the off-target. The oligo, on-target and off-target
strands are initially set at the same concentration $C_{in}$ and we define the duplexing score as: 

log( C_{oligo + off-t} /C_{oligo + off-t}  + C_{oligo + on-t}  ). 
    