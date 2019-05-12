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
            .setOutputCol("words")


        val word2vec = new Word2Vec()
              .setInputCol(tokenizer.getOutputCol)
              .setOutputCol("features")

        val rnd = new RandomForestClassifier()

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, word2vec, rnd))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
          .addGrid(rnd.maxBins, Array(28))
          .addGrid(rnd.maxDepth, Array(8))
          .addGrid(rnd.numTrees, Array(30, 100))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(trainValidationRatio)

        val model = trainValidationSplit.fit(trainDF)

        println( "Best score on validation set " + model.validationMetrics.max )

        model.getEstimatorParamMaps.zip(model.validationMetrics).foreach(println)

        (model.validationMetrics.max, "Random Forest", model)
    }
}
