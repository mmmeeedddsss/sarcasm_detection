import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType

object entry_point {

    val trainValidationSplitRatio = 0.8

    def main(args: Array[String]) {

        val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        val sqlContext = spark.sqlContext
        import sqlContext.implicits._
        val schema = ScalaReflection.schemaFor[Comment].dataType.asInstanceOf[StructType]
        val commentsDF = spark.read
            .option("header", false)
            .option("delimiter", "\t")
            .schema(schema)
            .csv("train-balanced-sarc.csv").as[Comment].filter( c => c.comment != null && c.comment != "" )

        commentsDF.printSchema()
        commentsDF.show()

        val dividedDatasets = commentsDF.randomSplit( Array(0.8, 0.2), seed = 1234 )
        val trainDF = dividedDatasets(0).cache()
        val testDF = dividedDatasets(1).cache()



        // More algortihms will be called here
        val tfidf_pair = tfidf.fit(trainDF, trainValidationSplitRatio)


        // Best algorithm on the validation set is selected and its score on test set is calculated in here
        val predictionsDF = tfidf_pair._2.transform(testDF)

        val evaluator = new BinaryClassificationEvaluator()
        println("On test data : " + evaluator.evaluate(predictionsDF))

        trainDF.show()
    }
}
