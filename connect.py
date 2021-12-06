from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext

# $SPARK_HOME/bin/spark-submit connect.py 2>log.txt

sc = SparkContext(appName="spam")
ssc = StreamingContext(sc, batchDuration= 3)
spark = SparkSession.builder.appName('spam').getOrCreate()
s=SQLContext(sc)
print('session started')

df=spark.readStream.format("socket").option("host","localhost").option("port","6100").load()
df.select("Subject").show()
