import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score


df = pd.read_csv('adult.data')
df.columns = ['age','workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

print(np.shape(df))
adult_test_df = pd.read_csv('/home/maneeshk/Downloads/adult/adult.test', skiprows=[0])
adult_test_df.columns = ['age','workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

df = pd.concat([df, adult_test_df])
print(np.shape(adult_test_df))

# getting statistics for each numeric column
print(df.describe())

# getting statistics for each categorical column
# getting count for all categorical columns

categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race','sex','native-country']
for colname in categorical_cols:
    print("="*50, colname)
    print(df[colname].value_counts())

# apply method
def replace_val(x):
    if x.strip() == '?': 
        x = 'other'
    return x.strip()

# setting ? values to other
df['workclass'] = df['workclass'].apply(replace_val)
df['occupation'] = df['occupation'].apply(replace_val)
df['native-country'] = df['native-country'].apply(replace_val)
# print(df['workclass'].unique())
# print(df['occupation'].unique())
# print(df['native-country'].unique())

# showing histogram to verify data distribution
# pt = df['age'].hist()
# pt = df['fnlwgt'].hist()
# pt = df['education-num'].hist()
# pt = df['capital-gain'].hist()
# pt = df['capital-loss'].hist()
# pt = df['hours-per-week'].hist()
# plt.show(pt)

# showing boxplot to verify data distribution
# pt = df.boxplot('age')
# pt = df.boxplot('fnlwgt')
# pt = df.boxplot('education-num')
# pt = df.boxplot('capital-gain')
# pt = df.boxplot('capital-loss')
# pt = df.boxplot('hours-per-week')
# plt.show(pt)

# showing bar chart for categorical feature
# value_counts() gives the series type so we have to convert that to dataframe for getting chart
# pt = df['workclass'].value_counts().to_frame() 
# pt = df['education'].value_counts().to_frame()
# pt = df['marital-status'].value_counts().to_frame()
# pt = df['occupation'].value_counts().to_frame()
# pt = df['relationship'].value_counts().to_frame()
# pt = df['race'].value_counts().to_frame()
# pt = df['sex'].value_counts().to_frame()
# pt = df['native-country'].value_counts().to_frame()
# pt = pt.plot.bar()
# plt.show(pt)

# getting missing values if column has
print ("getting null values if any")
for colname in df.columns:
    if df[colname].isnull().any():
        print colname


# getting dummy variables of all categorical data
dum1 = pd.get_dummies(df['workclass'], prefix='workclass', prefix_sep='_')
df1 = df.drop(['workclass'], axis=1)

# added new column in DF to assume that it has polynomial relation with age and hours-per-week
df1['age+work_hours'] = df['age'] * df['hours-per-week']

dum2 = pd.get_dummies(df['education'], prefix='education', prefix_sep='_')
df1 = df1.drop(['education'], axis=1)

dum3 = pd.get_dummies(df['marital-status'], prefix='marital-status', prefix_sep='_')
df1 = df1.drop(['marital-status'], axis=1)

dum4 = pd.get_dummies(df['occupation'], prefix='occupation', prefix_sep='_')
df1 = df1.drop(['occupation'], axis=1)

dum5 = pd.get_dummies(df['relationship'], prefix='relationship', prefix_sep='_')
df1 = df1.drop(['relationship'], axis=1)

dum6 = pd.get_dummies(df['race'], prefix='race', prefix_sep='_')
df1 = df1.drop(['race'], axis=1)

dum7 = pd.get_dummies(df['sex'], prefix='sex', prefix_sep='_')
df1 = df1.drop(['sex'], axis=1)

dum8 = pd.get_dummies(df['native-country'], prefix='native-country', prefix_sep='_')
df1 = df1.drop(['native-country'], axis=1)

df1 = pd.concat([df1, dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8],axis =1)
# print(df1.columns)

# getting corelation of numeric columns
print(df.corr())

# getting scatter plot to visulize the corelation
# pt = df.plot.scatter(x='age', y='hours-per-week')
# plt.show(pt)

# getting information abt dataframe
# print(df1['label'].head(5))

def class_lable(x):
    if x.strip()=='>50K' or x.strip()=='>50K.': 
        return 1 
    else: 
        return 0

# def new_age_as_category(x):
#     if x<=25:
#         return 0
#     elif x>25 and x<=50:
#         return 1
#     elif x>50 and x<=75:
#         return 2
#     else:
#         return 3

def new_age(x):
    if x<=20:
        return 15
    elif x>20 and x<=30:
        return (20+30)/2
    elif x>30 and x<=40:
        return (30+40)/2
    elif x>40 and x<=50:
        return (40+50)/2
    elif x>50 and x<=60:
        return (50+60)/2
    elif x>60 and x<=70:
        return (60+70)/2
    elif x>70 and x<=80:
        return (70+80)/2
    elif x>80 and x<=90:
        return (80+90)/2
    elif x>90 and x<=100:
        return (90+100)/2

# creating age as category rather number
# df1['age'] = df1['age'].apply(new_age_as_category)
# print(df1['age'].value_counts())

# creating age as new column
df1['age'] = df1['age'].apply(new_age)

# print(df1['age'].value_counts())


# apply method to convert the class label as 0 or 1
df1['label'] = df1['label'].apply(class_lable)
print(df1['label'].value_counts())

# print(df1.columns.get_loc('label'))
# delete the old label and added new at last of all columns
df1['new_label'] = df1['label']
df1 = df1.drop(['label'], axis=1)
# print(df1.head(5))

# normalize data
# scaler = StandardScaler()
# df1 = scaler.fit_transform(df1)
# df.columns = ['age','workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race','sex',
# 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

df1['age'] = (df1['age'] - df1['age'].min())/(df1['age'].max() - df1['age'].min())
df1['fnlwgt'] = (df1['fnlwgt'] - df1['fnlwgt'].min())/(df1['fnlwgt'].max() - df1['fnlwgt'].min())
df1['education-num'] = (df1['education-num'] - df1['education-num'].min())/(df1['education-num'].max() - df1['education-num'].min())
df1['capital-gain'] = (df1['capital-gain'] - df1['capital-gain'].min())/(df1['capital-gain'].max() - df1['capital-gain'].min())
df1['capital-loss'] = (df1['capital-loss'] - df1['capital-loss'].min())/(df1['capital-loss'].max() - df1['capital-loss'].min())
df1['hours-per-week'] = (df1['hours-per-week'] - df1['hours-per-week'].min())/(df1['hours-per-week'].max() - df1['hours-per-week'].min())
df1['age+work_hours']= (df1['age+work_hours'] - df1['age+work_hours'].min())/(df1['age+work_hours'].max() - df1['age+work_hours'].min())

# dividing training and test dataset
train_df = df1.iloc[0:32560,:]
test_df = df1.iloc[32560:,:]

# train_df = df1.iloc[0:26048,:]
# test_df = df1.iloc[26048:,:]

train_x = train_df.iloc[:,:-1]
train_y = train_df['new_label'].to_frame()

print(train_y['new_label'].value_counts())
new_train_y = []
for i in range(0, len(train_y)):
    new_train_y.append([train_y.iloc[i]['new_label']])

test_x = test_df.iloc[:,:-1]
test_y = test_df['new_label'].to_frame()

new_test_y = []
for i in range(0, len(test_y)):
    new_test_y.append([test_y.iloc[i]['new_label']])

# applying ML algos
print("LR")
lr = LogisticRegression()
# train_y = train_y.values
print(np.shape(train_x))
print(np.shape(new_train_y))
# print train_y
lr.fit(train_x, new_train_y)
print(lr.score(test_x, new_test_y))

# draw ROC curve and calculate AUC
# probs = lr.predict_proba(test_x)
# print(probs)
# probs = probs[:,1]
# print(probs)
# auc = roc_auc_score(test_y, probs)
# print('AUC: %.3f' % auc)
# # calculate roc curve
# fpr, tpr, thresholds = roc_curve(test_y, probs)
# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr, tpr, marker='.')
# # show the plot
# plt.show()

print("DT")
tr = DecisionTreeClassifier()
tr.fit(train_x, new_train_y)
print(tr.score(test_x, new_test_y))

print("NB")
nb = BernoulliNB()
nb.fit(train_x, new_train_y)
print(nb.score(test_x, new_test_y))

print("KNN")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, new_train_y)
print(knn.score(test_x, new_test_y))

print("NN")
nn = MLPClassifier(hidden_layer_sizes=(100,))
nn.fit(train_x, new_train_y)
print(nn.score(test_x, new_test_y))

print("Random Forest")
rf = RandomForestClassifier(n_estimators=100, max_depth=2)
rf.fit(train_x, new_train_y)
print(rf.score(test_x, new_test_y))

print("Ada Boost")
ab = AdaBoostClassifier()
ab.fit(train_x, new_train_y)
print(ab.score(test_x, new_test_y))

print("Bagging")
bg = BaggingClassifier()
bg.fit(train_x, new_train_y)
print(bg.score(test_x, new_test_y))

print("Gradieant Boost")
gb = GradientBoostingClassifier()
gb.fit(train_x, new_train_y)
print(gb.score(test_x, new_test_y))

print("Linear SVC")
svm = LinearSVC()
svm.fit(train_x, new_train_y)
print(svm.score(test_x, new_test_y))

# print("Kernel SVM")
# for kernel in ('linear', 'poly', 'rbf'):
#     print("kernel type==>", kernel)
#     clf = SVC(kernel=kernel, gamma=2)
#     clf.fit(train_x, new_train_y)
#     print(clf.score(test_x, new_test_y))