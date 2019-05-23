import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object LinearSVC extends ml_algorithm {

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("Linear SVC")

        val lr = new LinearSVC()

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, word2vec, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new RandomGridBuilder(1)
            .addDistr(lr.regParam, Array(0.01))
            .addDistr(word2vec.windowSize, Array(3,4,6))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            //.setTrainRatio(trainValidationRatio)

        val model = trainValidationSplit.fit(trainDF)
        println( "Best score on validation set " + model.validationMetrics.max )

        model.getEstimatorParamMaps
            .zip(model.validationMetrics).foreach( t => println(t) )

        (model.validationMetrics.max, "linearSVC", model)
    }
}
