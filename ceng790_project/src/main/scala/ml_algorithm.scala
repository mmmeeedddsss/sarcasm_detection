import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, StringIndexer, Tokenizer, Word2Vec}
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.Dataset

trait ml_algorithm {

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val tokenizer = new Tokenizer()
      .setInputCol("comment")
      .setOutputCol("words")

    val vectorizer = new HashingTF()
      .setInputCol("words")
      .setOutputCol("vectorized_comment")

    val idf = new IDF()
      .setInputCol(vectorizer.getOutputCol)
      .setOutputCol("features")

    val word2vec = new Word2Vec()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val evaluator = new BinaryClassificationEvaluator()

    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel)
}
