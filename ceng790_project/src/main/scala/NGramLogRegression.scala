import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.Dataset

object NGramLogRegression extends ml_algorithm {

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel) = {

        println("NGram LR")

        val indexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(trainDF)

        val tokenizer = new Tokenizer()
            .setInputCol("comment")
            .setOutputCol("words")

        val ngram = new NGram()
            .setInputCol("words")
            .setOutputCol("ngrams")

        val countVectorizer = new CountVectorizer()
            .setInputCol("ngrams")
            .setOutputCol("ngram_vectors")

        val lr = new LogisticRegression()
            .setMaxIter(20)
            .setLabelCol("indexedLabel")
            .setFeaturesCol("ngram_vectors")
            //.setElasticNetParam(0.8)

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, ngram, countVectorizer, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(lr.regParam, Array(0.1, 0.01))
            .addGrid(ngram.n, Array(2))
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
