#!/usr/bin/env python
# coding: utf-8

# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install --upgrade matplotlib')
get_ipython().system('pip install numpy')


# In[80]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('chatgpt1.csv',nrows=10000, encoding="ISO-8859-1")


# In[81]:


#preprocessing/data cleaning
# Replace None values in the 'Text' column with empty strings
data['Text'] = data['Text'].apply(lambda x: x if x is not None else '')


# In[82]:


vectorizer = TfidfVectorizer() #feature extraction
X = vectorizer.fit_transform(data['Text'])


# In[83]:


#user classification 
# Set target variable (user IDs) for classification
y = data['User']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# In[84]:


#engagement prediction
# Set target variable (retweet count) for engagement prediction
y = data['RetweetCount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# In[85]:


from sklearn.cluster import KMeans


n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(X)  # Fit the model and obtain cluster labels


# In[ ]:


clusters=kmeans.labels_


# In[ ]:


data['Cluster']=clusters


# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


import matplotlib.pyplot as plt

# Compute silhouette score
silhouette_avg = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Plot the elbow method
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[ ]:




