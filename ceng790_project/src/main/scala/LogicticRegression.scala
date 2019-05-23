import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object LogicticRegression extends ml_algorithm {

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("Logistic Regression with TF-IDF")

        val lr = new LogisticRegression()
            .setLabelCol("indexedLabel")
            .setFeaturesCol("features")

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, vectorizer, idf, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(lr.regParam, Array(0.01))
            .addGrid(lr.elasticNetParam, Array(0.07))
            .addGrid(lr.maxIter, Array(25))
            .addGrid(vectorizer.numFeatures, Array(524288))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(trainValidationRatio)

        val model = trainValidationSplit.fit(trainDF)
        println( "Best score on validation set " + model.validationMetrics.max )

        model.getEstimatorParamMaps
            .zip(model.validationMetrics).foreach( t => println(t) )

        (model.validationMetrics.max, "tf-idf", model)
    }
}
