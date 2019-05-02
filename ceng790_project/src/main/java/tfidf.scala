import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType

object tfidf {

    // define the Credit Schema
    case class Comment(
        label: Int, comment: String, author: String, subreddit: String, score: Int,
        ups: Int, downs: Int, date: String, created_utc: String, parent_comment: String
     )

    def main(args: Array[String]) {

        val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
        val sc = spark.sparkContext
        val sqlContext = spark.sqlContext
        import sqlContext.implicits._
        val schema = ScalaReflection.schemaFor[Comment].dataType.asInstanceOf[StructType]
        val commentsDF = spark.read
            .option("header", false)
            .option("delimiter", "\t")
            .schema(schema)
            .csv("train-balanced-sarc.csv").as[Comment].filter( c => c.comment != null && c.comment != "" )

        val commentsRdd = commentsDF.rdd
        commentsDF.printSchema()
        commentsDF.show()

        // PART 3
        val dividedDatasets = commentsDF.randomSplit( Array(0.8, 0.2), seed = 1234 )
        val trainDF = dividedDatasets(0).cache()
        val testDF = dividedDatasets(1).cache()

        val indexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(trainDF)

        val tokenizer = new Tokenizer()
            .setInputCol("comment")
            .setOutputCol("words")

        val vectorizer = new HashingTF()
            .setInputCol(tokenizer.getOutputCol)
            .setOutputCol("vectorized_comment")

        val idf = new IDF()
            //.setMinDocFreq(params.minDocFreq)
            .setInputCol(vectorizer.getOutputCol)
            .setOutputCol("features")


        val lr = new LogisticRegression()
            .setMaxIter(10)
            //.setRegParam(0.3)
            //.setElasticNetParam(0.8)

        val mlPipeline = new Pipeline()
            .setStages(Array(indexer, tokenizer, vectorizer, idf, lr))

        val evaluator = new BinaryClassificationEvaluator()

        val paramGrid = new ParamGridBuilder()
            .addGrid(lr.elasticNetParam, Array(0.4, 0.6, 0.8, 0.9))
            .build()

        val trainValidationSplit = new TrainValidationSplit()
            .setEstimator(mlPipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(0.8)

        val model = trainValidationSplit.fit(trainDF)

        model.getEstimatorParamMaps
            .zip(model.validationMetrics).foreach( t => println(t) )

        val predictionsDF = model.transform(testDF)

        println("On test data : " + evaluator.evaluate(predictionsDF))
    }
}
