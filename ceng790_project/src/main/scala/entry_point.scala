import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType

object entry_point {

    val trainValidationSplitRatio = 0.8

    def main(args: Array[String]) {

        val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
        val sc = spark.sparkContext
        sc.setLogLevel("WARN")

        val sqlContext = spark.sqlContext
        import sqlContext.implicits._
        val schema = ScalaReflection.schemaFor[Comment].dataType.asInstanceOf[StructType]
        val commentsDF = spark.read
            .option("header", false)
            .option("delimiter", "\t")
            .schema(schema)
            .csv("train-balanced-sarc.csv").as[Comment]
            .filter( c => c.comment != null && c.comment != "")
            //.map( c => helpers.include_subreddit(c) )

        commentsDF.printSchema()
        //commentsDF.show()

        val dividedDatasets = commentsDF.randomSplit( Array(0.8, 0.2), seed = 1234 )
        val trainDF = dividedDatasets(0).cache()
        val testDF = dividedDatasets(1).cache()



        // More algortihms will be added here
        val trained_model_tuples = Array(
            //LogicticRegression.fit(trainDF, trainValidationSplitRatio)
            //word2vec.fit(trainDF, trainValidationSplitRatio)
            //NaiveBayes.fit(trainDF, trainValidationSplitRatio),
            //LinearSVC.fit(trainDF, trainValidationSplitRatio),
            RandomForest.fit(trainDF, trainValidationSplitRatio)
        )

        // Best algorithm on the validation set is selected and its score on test set is calculated in here
        val best_model = trained_model_tuples.maxBy( m => m._1 )
        val predictionsDF = best_model._3.transform(testDF)

        val evaluator = new BinaryClassificationEvaluator()
        println("Best model %s on test data : %f".format(best_model._2, evaluator.evaluate(predictionsDF)))
    }
}
