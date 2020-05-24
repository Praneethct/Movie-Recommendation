import pyspark
import math
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS


conf = SparkConf()
sc = SparkContext(conf = conf)



movie = sc.textFile("./ml-latest-small/movies.csv")
movie_header = movie.take(1)[0]
movies = movie.filter(lambda l : l != movie_header).map(lambda l : l.split(",")).map(lambda l : (int(l[0]), str(l[1])))
print(movie_header)
print(movies.take(3))

m = dict(movies.collect())

rating = sc.textFile("./ml-latest-small/ratings.csv")
rating_header = rating.take(1)[0]
ratings = rating.filter(lambda l : l != rating_header).map(lambda l : l.split(",")).map(lambda l : (int(l[0]), int(l[1]), float(l[2])))
print(rating_header)
print(ratings.take(3))



train_data, test_data = ratings.randomSplit([0.8, 0.2])
test = test_data.map(lambda l : (l[0], l[1]))



rec = ALS.train(train_data, 8, seed = 5, iterations = 10, lambda_ = 0.1)
pred = rec.predictAll(test).map(lambda l : ((l[0], l[1]), l[2]))
rating_pred = test_data.map(lambda l : ((l[0], l[1]), l[2])).join(pred)



RMSE = math.sqrt(rating_pred.map(lambda l : (l[1][0] - l[1][1])**2).mean())
print("RMSE : ", RMSE)


MAE = rating_pred.map(lambda l: abs(l[1][0] - l[1][1])).mean()
print("MAE : ", MAE)


user_id = input("enter user id : ")

user_recommendations = rec.recommendProducts(int(user_id), 10)

r = []
for i in user_recommendations:
    r.append(i[1])


rm = []
for i in r:
    rm.append(m[i])

for i in rm:
    print(i)
