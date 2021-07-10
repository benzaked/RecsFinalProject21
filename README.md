
# Introduction
This repo contains an Attempt for improvement
Neural-Attentive-Item-Similarity-Model based on [this](https://ieeexplore.ieee.org/abstract/document/8352808) paper and [this](https://github.com/hegongshan/neural_attentive_item_similarity) repository
as part of RecSys curse.   
### Requirements

* Python 3

* TensorFlow 2.0+

* NumPy (latest version)

* SciPy (latest version)

* jupyterlab

### Training NAIS :

```
python NAIS.py --pretrain 1 --path data --data_set_name ml-1m --epochs 100 --num_neg 4 --embedding_size 16 --lr 0.01
```

### Getting NAIS prediction:
```
python Get_Nais_Predictions.py --data_set_name ml-1m --embedding_size 16 
```
This command will use the pre-trained NAIS model and create predictions to the folder:
```
/predictions/NAIS/ml-1m/ 
```

### Creating explainable recommendations:
This class contains User-User, Item-Item, content-based and popularity recommendations systems.
By executing the following command
```
python RecommendationsAlgorithms.py  
```
The script will create 3 files- item_item.csv, user-user.csv and output.csv
those files contains the recommendations according to each model and division of each item recommended by NAIS into the appropriate algorithm for every user  in the dataset  in the output.csv file 
### Reorder 
```
python RecommendationsAlgorithms.py  
```
This script will re order according to our approaches. The script takes the recommendations 
provided in the output.csv and create new file 're_ordered_{embedding_size.csv} '
Both files will save in :
```
/predictions/{dataset}/
```

to generate the plots please see the files
```
top@k predictions all models compare.ipynb
top_k_predictions_all_models_compare.ipynb
```
