from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("nlp_nb").getOrCreate()
df=spark.read.csv('spam/test.csv', inferSchema=True, header=True)
#sep="\t"



#clean data
from pyspark.sql.functions import length
df=df.withColumn("length",length(df['Message']))
df.groupby("Spam/Ham").mean()

#Esemblers/Feature Transformation

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer

tokenizer= Tokenizer(inputCol="Message", outputCol="token_tokens")
stop_word_remover = StopWordsRemover(inputCol="token_text", outputCol="stop_token")
count_vec = CountVectorizer( inputCol="stop_token", outputCol="c_vec")
idf= IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer( inputCol ="Spam/Ham", outputCol="label")

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

cleaned = VectorAssembler(inputCols=['tf_idf','length'], outputCol="features")

#Model- Naive Bayes Model

from pyspark.ml import Pipeline
pipeline= Pipeline(stages=[
				ham_spam_to_num,
				tokenizer,
				stop_word_remover,
				count_vec,
				idf,
				cleaned
])

cleaner = pipeline.fit(df)
clean_df = cleaner.transform(df)

#Train Model on train.csv file

clean_df = clean_df.select(['label','feature'])

pred= nb.fit(clean_df)


##### evaluation of testing part on test.csv 

res = pred.transform(pred)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

eval = MulticlassClassificationEvaluator()
acc = eval.evaluate(res)
print(f"Accuracy: {acc * 100}")
