#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(2)


# In[6]:


movies.shape


# In[7]:


credits.head()


# In[8]:


movies = movies.merge(credits,on='title')


# In[9]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[10]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head()


# In[12]:


import ast


# In[13]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[16]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[17]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L


# In[18]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[20]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[21]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[22]:


movies.sample(5)


# In[23]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[24]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[25]:


movies.head()


# In[26]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[27]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[28]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[29]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[31]:


vector = cv.fit_transform(new['tags']).toarray()


# In[32]:


vector.shape


# In[33]:


from sklearn.metrics.pairwise import cosine_similarity


# In[34]:


similarity = cosine_similarity(vector)


# In[35]:


similarity


# In[36]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[37]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[38]:


recommend('Gandhi')


# In[39]:


import pickle


# In[40]:


pickle.dump(new,open('movie_list.pkl','wb'))


# In[41]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[45]:


pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))


# In[44]:





# In[46]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




