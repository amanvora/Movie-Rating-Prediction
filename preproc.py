#####################################################
# Date      :       12th Dec 2016                   #
# Name      :       Aman Vora                       #
# USC ID    :       4057796804                      #
# Email     :       amanvora@usc.edu                #
# Written for EE660 final project                   #
#						    #
# This script reads the database and performs	    #
# 3 preprocessing steps:			    #
# 1. Handle missing data of numerical features	    #
# 2. Convert categorical to numerical values        #
# 3. Standaradize each feature			    #
#####################################################

# Import dependencies and/or other modules
import pandas as pd
from sets import Set
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import scikit-learn modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale

# Load database
D_orig = pd.read_csv('movie_metadata.csv');

D = D_orig
# Remove features/columns not to be used:
del D['director_name']
del D['actor_2_name']
del D['actor_1_name']
del D['movie_title']
del D['actor_3_name']
del D['plot_keywords']
del D['movie_imdb_link']

#Fill in missing data of numerical columns
numericalCols = list(D.columns)
categoricalCols = ['genres','color','language','country','content_rating']
for i in categoricalCols:
    numericalCols.remove(i)
D[numericalCols] = D[numericalCols].fillna(D[numericalCols].mean())
print ('Missing data of numerical features filled')

# Parse genres to make as a list
for g in range(len(D['genres'])):
    D['genres'][g] = D['genres'][g].split('|')
DGen = list(D['genres'])
genres = list(DGen[0])
for g in range(1,len(DGen)):
    genres.extend(DGen[g])
genres = list(sorted(Set(genres)))
vec_data = pd.DataFrame(index = D.index, columns = genres)
for i in vec_data.index:
    currGen = D['genres'][i]
    for c in vec_data.columns:
        vec_data[c][i] = 1 if c in currGen else 0

D = D.drop('genres', axis=1)
D = D.join(vec_data)

D = D.dropna()

###-------------Uncomment the following lines to view the---------------------#
###--------------categories in each categorical feature-----------------------#
##print ('List of genres')
##print genres
##print len(genres)
##print '\n'
##
##print ('List of languages')
##DLangs = D['language']
##langs = Set(DLangs)
##print sorted(langs)
##print len(langs)
##print '\n'
##
##print ('List of countries')
##DCntry = D['country']
##countries = Set(DCntry)
##print sorted(countries)
##print len(countries)
##print '\n'
##
##print ('List of content rating')
##DContRate = D['content_rating']
##contRate = Set(DContRate)
##print sorted(contRate)
##print len(contRate)
##print '\n'
#---------------------------------------------------------------------------#

# Convert 'colsToVect' categorical features to numerical using one hot encoding
vec = DictVectorizer()
colsToVect = ['color','language','country','content_rating']
vec_data = pd.DataFrame(vec.fit_transform(D[colsToVect].to_dict(outtype='records')).toarray())
vec_data.columns = vec.get_feature_names()
vec_data.index = D.index
D = D.drop(colsToVect, axis=1)
D = D.join(vec_data)
print ('Converted categorical data to numerical')
# Ensure output is last column
dImdb = D['imdb_score']
D = D.drop('imdb_score', axis=1)
D = D.join(dImdb)
newAttr = list(D.columns)
D_Arr = np.array(D)
D_Arr = D_Arr.astype(np.float)
imdbScores = D_Arr[:,-1]

# Standardize each feature
D_Arr = scale(D_Arr)
print ('Standardized each feature')

# Save workspace variable to a file
np.savetxt('preprocessed.txt',D_Arr)
print ('Generated text file of preprocessed dataset')

np.savetxt('imdbScores.txt',imdbScores)
print ('Generated text file of preprocessed ground truth')
