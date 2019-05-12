import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object RandomForest extends ml_algorithm {


    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("Random Forest")

        val indexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(trainDF)

        val tokenizer = new Tokenizer()
            .setInputCol("comment")
            .setOutputCol("stopWords")

        val remover = new StopWordsRemover()
          .setInputCol("stopWords")
          .setOutputCol("words")

        val word2vec = new Word2Vec()
              .setInputCol(remover.getOutputCol)
              .setOutputCol("features")

        val rnd = new RandomForestClassifier()

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, remover, word2vec, rnd))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
          .addGrid(rnd.maxBins, Array(28, 40))
          .addGrid(rnd.maxDepth, Array(5, 12))
          .addGrid(rnd.numTrees, Array(100))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(trainValidationRatio)

        val model = trainValidationSplit.fit(trainDF)
        println("stop Words is used")
        println( "Best score on validation set " + model.validationMetrics.max )

        model.getEstimatorParamMaps.zip(model.validationMetrics).foreach(println)

        (model.validationMetrics.max, "Random Forest", model)
    }
}
