Libraries required:
1. Scikit-learn
2. Matplotlib
3. numpy



Instructions to run:
1. Please run preproc.py first. It will generate 'preprocessed.txt' and 'imdbScores.txt' to be used by the other two scripts.

2. modelSelCV.py and preproc.py each have commented sections in order to view plot of covariance matrix and categorical features respectively.

3. Since model selection performs an exhaustive evaluation of all the models across all the hyperparamters, it takes a long time to execute (approx. 30 minutes depending on the system)




Code referred:
1. For one-hot enconding of catgorical features:
http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

2. For splitting the data to (train,test) and loading the database
https://www.kaggle.com/nsnilay/d/deepmatrix/imdb-5000-movie-dataset/predicting-imdb-ratings-using-numerical-attributes/notebook