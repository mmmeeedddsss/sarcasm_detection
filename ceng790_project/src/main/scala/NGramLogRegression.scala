import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object NGramLogRegression extends ml_algorithm {

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("NGram LR")

        val ngram = new NGram()
            .setInputCol("words")
            .setOutputCol("ngrams")

        val vectorizer = new HashingTF()
            .setInputCol("ngrams")
            .setOutputCol("vectorized_comment")

        val idf = new IDF()
            .setInputCol(vectorizer.getOutputCol)
            .setOutputCol("features")

        val lr = new LogisticRegression()
            .setMaxIter(20)
            .setLabelCol("indexedLabel")
            .setFeaturesCol("features")
            //.setElasticNetParam(0.8)

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, ngram, vectorizer, idf, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(lr.regParam, Array(0.01))
            .addGrid(lr.elasticNetParam, Array(0.08))
            .addGrid(lr.maxIter, Array(20, 25))
            .addGrid(ngram.n, Array(2, 3))
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

        (model.validationMetrics.max, "NGram-LR", model)
    }
}
