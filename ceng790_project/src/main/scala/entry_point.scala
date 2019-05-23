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
            .filter( _.comment != null )
            //.map( c => helpers.text_clean(c) )
            .filter( _.comment.replaceAll(" ", "") != "")

        val precalculation = commentsDF.rdd.map( c => c.comment.split(" ").map( w => (w,1) )  ).flatMap( r => r.toList )

        val wordFrequencies = precalculation.combineByKey( c1 => c1 ,(c1 : Int, c2: Int) => c1+c2,  (c1 : Int, c2: Int) => c1+c2 )
        val wordsDS = wordFrequencies.filter(r => r._2 > 10)
          .map( r => r._1 ).map( c => Comment(0,c,"","",null,null,null,"","","") ).toDS()

        commentsDF.printSchema()
        //commentsDF.show()

        val dividedDatasets = commentsDF.randomSplit( Array(0.8, 0.2), seed = 1234 )
        val trainDF = dividedDatasets(0).cache()
        val testDF = dividedDatasets(1).cache()

        // Some Statistics
        //helpers.countSarcastics(trainDF)

        // More algortihms will be added here
        val trained_model_tuples = Array(
            LogicticRegression.fit(trainDF, trainValidationSplitRatio)
            //word2vec.fit(trainDF, trainValidationSplitRatio),
            //NaiveBayes.fit(trainDF, trainValidationSplitRatio),
            //LinearSVC.fit(trainDF, trainValidationSplitRatio),
            //RandomForest.fit(trainDF, trainValidationSplitRatio),
            //NGramLogRegression.fit(trainDF, trainValidationSplitRatio)
        )

        // Best algorithm on the validation set is selected and its score on test set is calculated in here
        val best_model = trained_model_tuples.maxBy( m => m._1 )
        val predictionsDF = best_model._3.transform(testDF)

        val evaluator = new BinaryClassificationEvaluator()
        println("Best model %s on test data : %f".format(best_model._2, evaluator.evaluate(predictionsDF)))

        if(false){
            import org.apache.spark.sql.functions._
            import org.apache.spark.ml.linalg.Vector
            val to_array1 = udf((v: Vector) => v.toDense.values)
            val to_array2 = udf((v: Vector) => v.toDense.values)
            best_model._3.bestModel.transform(wordsDS).orderBy(to_array1($"rawPrediction").getItem(0).desc).limit(15)
              .select("comment", "rawPrediction", "probability", "indexedLabel", "prediction" ).show(15, false)

            val to_array = udf((v: Vector) => v.toDense.values)
            best_model._3.bestModel.transform(wordsDS).orderBy(to_array2($"rawPrediction").getItem(1).desc).limit(15)
              .select("comment", "rawPrediction", "probability", "indexedLabel", "prediction" ).show(15, false)
        }

        println("....")
        while(true){
            val text = scala.io.StdIn.readLine()
            if(text == "" || text == "\n")
                return
            val userIDataSet = Seq( Comment(0,text,"","",null,null,null,"","","") ).toDS()
            val prediction = best_model._3.bestModel.transform(userIDataSet).select("prediction").first().getDouble(0)
            if ( prediction == 0.0 )
              println("Not Sarcastic")
            else
              println("Sarcastic")
        }
    }
}
