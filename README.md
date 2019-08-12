# Sarcasm Detection using Spark
#### Mert Tunç, Egemen Berk Galatalı

<br/> 

1.3 million reddit comments that is labeled as sarcastic or not is used as dataset to create a sarcasm classifier.
Several methods for preprocessing, feature extraction and ml models are combined to get the best results. Implementation is done using Scala with Spark.

77% accurcacy on test set is taken with the best preprocessing - feature extraction - ml model selection - semi optimized hyper parameters combination. Please note that no other coloumns than comments itselves and labels are used on training or testing.
