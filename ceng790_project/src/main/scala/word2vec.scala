import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object word2vec extends ml_algorithm {

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("Word2Vec")

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

        val lr = new LogisticRegression()
            .setMaxIter(10)

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, word2vec, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid( word2vec.windowSize, Array(5) )
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

        (model.validationMetrics.max, "word2vec", model)
    }
}
