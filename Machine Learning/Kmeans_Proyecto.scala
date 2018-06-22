/////////////////////////////////
// K MEANS PROYECTO/EJERCICIO ////
///////////////////////////////

// En este proyecto se agruparán los clientes de un distribuidor mayorista
// basado en las ventas de algunas categorías de productos

// Fuente de los datos
//http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

// Aquí está la información sobre los datos:
// 1) FRESCO: gasto anual en productos frescos;
// 2) LECHE: gasto anual en productos lácteos;
// 3) SUPERMERCADO: gasto anual en productos comestibles;
// 4) CONGELADO: gasto anual en productos congelados
// 5) DETERGENTES: gasto anual en detergentes y productos de papel
// 6) DELICATESSEN: gasto anual en productos delicatessen
// 7) CANAL: canal de clientes - Horeca (Hotel / Restaurante / Cafe) o canal de venta minorista (Nominal)
// 8) REGIÓN: clientes Región- Lisnon, Oporto u Otro (Nominal)

//DAtos originales en Inglés:

// Here is the info on the data:
// 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
// 2)	MILK: annual spending (m.u.) on milk products (Continuous);
// 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
// 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
// 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
// 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
// 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
// 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)


// Importamos SparkSession
import org.apache.spark.sql.SparkSession

//Usamos el sig. código para tener un reporte del Error
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creamos una nueva sesión (instancia) de Spark
val spark = SparkSession.builder().getOrCreate()

// Importamos Kmeans clustering (Algorithmo)
import org.apache.spark.ml.clustering.KMeans

// Se importan los datos
val datos = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

// Seleccionamos las siguientes columnas para el conjunto de entrenamiento:
// Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
// LLamamos a esta nueva característica carac_datos
val carac_datos = datos.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")


// Importamos VectorAssembler y Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Creamos un nuevo objeto VectorAssembler al que llamaremos asemblando pora la característica
// Se configura la salida para que sea la columna características
// NO HAY RESPUESTAS/LABERLS/ETIQUETAS
val asemblando = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("caracteristicas")

// Usamos assembler para transformar carac_datos
// Llamamos a estos datos nuevos datos_ent
val datos_ent = asemblando.transform(carac_datos).select("caracteristicas")

// Creamos un modelo Kmeans con K=3
val kmeans = new KMeans().setK(3).setSeed(1L)

// Entrenamos el modelo con el conjunto de entrenamiento
val model = kmeans.fit(datos_ent)

// Es el turno de saber que tan bien se comportó gracias al error
val WSSSE = model.computeCost(datos_ent)
println(s"Error (suma de errores cuadráticos) = $WSSSE")

// Mostramos el resultado
println("Clusters (Centros): "
model.clusterCenters.foreach(println)
