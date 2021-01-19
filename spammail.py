import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# load the dataset to pandas Data Frame
raw_mail_data = pd.read_csv('spam.csv',encoding = 'ISO-8859-1')
# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

mail_data.shape
mail_data.head() #sample data

# label spam mail as 0; Non-spam mail (ham) mail as 1
mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1

# separate the data as text and label. X--> text; Y-->label
X = mail_data['v2']
Y = mail_data['v1']

# split the data as train data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=3)


# transform the text data to feature vectors that can be used as input to the SVM model using TfidfVectorizer
# convert the text to lower case letters
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


#convert Y_train and Y_test values are integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#training the SVM model with training data
model = LinearSVC()
model.fit(X_train_features,Y_train)

# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)

print("Accuracy on training data :",accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

print("Accuracy on testing data :",accuracy_on_test_data)

input_mail = ["WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.,,,"]
# convert text  to feature vectors
input_mail_features =feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_mail_features)
print(prediction)

if(prediction[0]==1):
  print('HAM MAIL')
else:
  print('SPAM MAIL')
