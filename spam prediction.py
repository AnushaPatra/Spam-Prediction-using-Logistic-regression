#!/usr/bin/env python
# coding: utf-8

# In[235]:


#RA2111027010022-Anusha Patra
#Q1 load the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# In[236]:


#connecting data into pandas
spam_database = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\spam.csv", encoding='latin1')
print(spam_database.head())


# In[237]:


#Q2 Preprocess the data
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in  stopwords.words('english')]
    return " ".join(text)


# In[238]:


# No. of rows and columns
spam_database.shape


# In[239]:


#No. of cells having NULL values
spam_database.isnull().sum()


# In[240]:


#Q3 Applying label encoding for string attributes
# Correct the column names to match your dataset
columns_to_encode = ["v1"]
encoded_data = pd.DataFrame()

for column in columns_to_encode:
    label_encoder = LabelEncoder()
    encoded_col = label_encoder.fit_transform(spam_database[column])
    encoded_data[column] = encoded_col
    print(encoded_col)


# In[241]:


# Calculate the count of spam and ham emails based on the encoded labels
spam_count = (encoded_data["v1"] == 0).sum()
ham_count = (encoded_data["v1"] == 1).sum()

print("Number of spam emails:", spam_count)
print("Number of ham emails:", ham_count)


# In[242]:


print(spam_database.columns)


# In[243]:


X = spam_database['v2']  # Use 'v2' as the column name for text messages
Y = encoded_data['v1']  # Use 'v1' as the column name for labels

print(X)


# In[244]:


print(Y)


# In[245]:


#Q4 split the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[246]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[247]:


# Separate ham and spam messages
ham_messages = spam_database[spam_database['v1'] == 'ham']['v2']
spam_messages = spam_database[spam_database['v1'] == 'spam']['v2']


# In[248]:


#Q5 plot wordclouds for ham and spam
ham_wordcloud = WordCloud(width=800, height=400).generate(" ".join(ham_messages))
plt.figure(figsize=(10, 5))
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Ham Messages')
plt.axis('off')
plt.show()


# In[249]:


spam_wordcloud = WordCloud(width=800, height=400).generate(" ".join(spam_messages))
plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Spam Messages')
plt.axis('off')
plt.show()


# In[252]:


#Q6 calculate tf and idf values
# Preprocess text column
spam_database['v2'] = spam_database['v2'].apply(text_preprocess)

# Combine preprocessed text data into a single list
documents = spam_database['v2'].tolist()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents to calculate TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Calculate TF-IDF values
tfidf_values = tfidf_matrix.toarray()

# Extract the feature names (terms)
terms = tfidf_vectorizer.get_feature_names_out()

# Create a DataFrame to display TF-IDF values
tfidf_df = pd.DataFrame(tfidf_values, columns=terms)


# In[253]:


# Display the TF-IDF values
print("TF-IDF Values:")
print(tfidf_df)


# In[254]:


#Q7 Calculate TF * IDF values
tf_idf_product = tfidf_values * np.array(tfidf_vectorizer.idf_)

# Create a DataFrame to display TF * IDF values
tf_idf_product_df = pd.DataFrame(tf_idf_product, columns=terms)

# Display the TF * IDF values
print("\nTF * IDF Values:")
print(tf_idf_product_df)


# In[255]:


#Q8 apply logistic regression
X = tfidf_values
Y = spam_database['v1']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)

# Predict on the testing set
Y_pred = logistic_regression.predict(X_test)


# In[256]:


#Q9 calculate the accuracy and F1 score along with confusion matrix
confusion = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Display the model evaluation
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[257]:


#Q10 Define the ham and spam features and define the spam messages in test test
# Identify ham and spam features
ham_features = X_test[Y_test == 'ham']
spam_features = X_test[Y_test == 'spam']

# Identify spam messages in the test set
spam_messages = spam_database.loc[spam_database['v1'] == 'spam', 'v2'].values
spam_messages_in_test = X_test[Y_pred == 'spam']

# Display ham and spam features
print("Ham Features:", ham_features)
print("Spam Features:", spam_features)

# Display spam messages in the test set
print("Spam Messages in Test:")
for msg in spam_messages_in_test:
    print(msg)


# In[ ]:





# In[ ]:




