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

        val swr = new StopWordsRemover()
            .setInputCol("words")
            .setOutputCol("cleaned_words")

        val vectorizer = new HashingTF()
            .setInputCol("cleaned_words")
            .setOutputCol("vectorized_comment")

        val idf = new IDF()
            //.setMinDocFreq(params.minDocFreq)
            .setInputCol(vectorizer.getOutputCol)
            .setOutputCol("features")

        val rnd = new RandomForestClassifier()
            .setSeed(1234)
            .setFeatureSubsetStrategy("auto")

        val mlPipeline = new Pipeline()
            .setStages(Array(tokenizer, swr, vectorizer, idf, rnd))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(rnd.maxBins, Array(25, 28 ))
            .addGrid(rnd.maxDepth, Array(4, 8))
            .addGrid(rnd.impurity, Array("gini"))
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
