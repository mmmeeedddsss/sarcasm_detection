import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.Dataset

trait ml_algorithm {
    def fit(trainDF : Dataset[Comment], trainValidationRatio: Double): (Double, String, TrainValidationSplitModel)
}
