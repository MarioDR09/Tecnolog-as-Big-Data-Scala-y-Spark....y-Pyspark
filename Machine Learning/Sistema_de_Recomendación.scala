import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

val ratings = spark.read.option("header","true").option("inferSchema","true").csv("/movie_ratings.csv")

ratings.head()
ratings.printSchema()

val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Construimos el modelo de recomendación usando ALS en los datos de entrenamienro
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model = als.fit(training)

// Evaluamos el model claculando el error promedio del rating real
val predic = model.transform(test)

// Importamos y utilizamos abs()
import org.apache.spark.sql.functions._
val error = predic.select(abs($"rating"-$"predicciones"))

// Eliminamos los NaNs
error.na.drop().describe().show()
