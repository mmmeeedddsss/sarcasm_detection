import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object RandomForest extends ml_algorithm {


    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("Random Forest")

        val rnd = new RandomForestClassifier()
            .setSeed(1234)
            .setFeatureSubsetStrategy("auto")

        val mlPipeline = new Pipeline()
            .setStages(Array(tokenizer, word2vec, rnd))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(rnd.maxBins, Array(25))
            .addGrid(rnd.maxDepth, Array(10))
            .addGrid(rnd.impurity, Array("entropy"))
            .addGrid(rnd.numTrees, Array(30))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)

        val model = trainValidationSplit.fit(trainDF)

        println( "Best score on validation set " + model.validationMetrics.max )

        model.getEstimatorParamMaps.zip(model.validationMetrics).foreach(println)

        (model.validationMetrics.max, "Random Forest", model)
    }
}
