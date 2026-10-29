import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#logistic Regression
X_large = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
                  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

# New labels (Notice the noise: some low hours passed, some high hours failed)
y_large = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 
                  1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X_large))  # Shuffle 0-19
train_idx = shuffled_indices[:14]
test_idx = shuffled_indices[14:]
X_train,X_test,y_train,y_test = train_test_split(X_large,y_large,test_size=0.2,random_state=42)
X_train = X_large[train_idx]
y_train = y_large[train_idx]
X_test = X_large[test_idx ]
y_test = y_large[test_idx ]
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)
model =LogisticRegression()
model.fit(X_train_reshaped,y_train)
predictions = model.predict(X_test_reshaped)



theta0_test = 0
theta1_test = 0
alpha =0.1
for i in range(1000):
    z = theta0_test + (theta1_test * X_train)
    p = 1 / (1 + np.exp(-z))  # Sigmoid
    p_clipped = np.clip(p, 1e-9, 1 - 1e-9) 
    m =len(X_train)
    loss_logistic = -(1/m) * np.sum(y_train * np.log(p_clipped) + (1-y_train) * np.log(1-p_clipped))
    a0 = np.mean(p_clipped-y_train)
    a1=np.mean((p_clipped-y_train)*X_train)
    theta0_test = theta0_test - alpha * a0
    theta1_test = theta1_test - alpha * a1
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss_logistic:.4f}, theta0 = {theta0_test:.4f}, theta1 = {theta1_test:.4f}")

print(f"Final Loss: {loss_logistic}")





z_train = theta0_test + theta1_test * X_test
p_train = 1 / (1 + np.exp(-z_train))

pred_train = (p_train >= 0.5).astype(int)

accuracy = np.mean(pred_train == y_test)
cor_p = np.sum(pred_train == y_test)
print(f"Test accuracy: {accuracy:.1%}") 
accuracy = np.mean(predictions == y_test)
print(f"Scikit-Learn Test Accuracy: {accuracy:.1%}")
print("Test size : 6 Correct predictions",cor_p )
print(pred_train)
print(predictions)
print(np.array_equal(pred_train, predictions))
cm = confusion_matrix(y_test, pred_train)
print(cm)

tp = 0
tn = 0
fp = 0
fn = 0
for  actual, predicted in zip(y_test,pred_train):
    if actual == 0 and predicted == 0:
        tn +=1
    elif actual == 1 and predicted == 1:
        tp +=1
    elif actual == 0 and predicted == 1:
        fp +=1
    elif  actual == 1 and predicted == 0:
        fn +=1
confusion_matrix =np.array([[tn,fp],
                           [fn,tp]])
print(confusion_matrix)
if (tp+ fp) == 0:
 Precision=0
else:
 Precision = tp/ (tp+fp)
 print(Precision)
if (tp+fn)== 0:
   recall =0
else: 
 recall =tp/(tp+fn)
 print(recall)
if (Precision + recall) == 0:
   f1_score =0
else: 
 f1_score = 2 *(Precision *recall) / (Precision + recall)
 print(f1_score)
print(tn+tp+fp+fn)
plt.scatter(X_train,y_train)
plt.plot(X_test,p_train)
plt.show()


    #Gda Classifier

class0 = [2, 2]
class1 =[5, 5]
cov = [[1.0, 0.5], [0.5, 1.0]]
samples=np.random.multivariate_normal(class0, cov, size=100,)
samples1=np.random.multivariate_normal(class1, cov, size=100,)
zeros = np.zeros(100)
ones= np.ones(100)
X = np.vstack((samples,samples1))
y = np.hstack((zeros, ones))
y_t = y.T
plt.scatter(X,y_t)
plt.show()
m = len(y)
predictions=[]
phi = 1/m *(y==1).sum()
mean0 = ((y == 0)[:, np.newaxis] * X).sum(axis=0) / (y == 0).sum()
mean1=((y == 1)[:, np.newaxis] * X).sum(axis=0) / (y == 1).sum()
means = np.array([mean0,mean1])
X_centered = np.copy(X)
X_centered[y == 0] = X_centered[y == 0] - mean0
X_centered[y == 1] = X_centered[y == 1] - mean1
cov_sum = X_centered.T @ X_centered
Sigma = (1 / m) * cov_sum
x_sample = X[0]
d=len(x_sample)
x_class0 = X[y == 0, 0]
y_class0 = X[y == 0, 1]

x_class1 = X[y == 1, 0]
y_class1 = X[y == 1, 1]

plt.scatter(x_class0, y_class0, color='blue', label='Class 0')
plt.scatter(x_class1, y_class1, color='red', label='Class 1')

def gaussian(x_sample,mean,Sigma):
    diff =  x_sample -mean
    denominator = (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(Sigma))
    constant = 1/denominator
    inner_math = -0.5 * (diff.T @ np.linalg.inv(Sigma) @ diff)
    probablity_density = constant*np.exp(inner_math)
    return probablity_density

probablity_density = gaussian(x_sample,mean0, Sigma)
print(probablity_density)
for x in X:
    dd = gaussian(x,mean0,Sigma)
    ff = gaussian(x,mean1,Sigma)
    score0 = dd * (1- phi)
    score1= ff * phi
    if score0 > score1:
     print("model predicts class 0")
     predictions.append(0)  # Append class label 0
    else:
     print("model predicts class 1")
     predictions.append(1)
accuracy=np.mean(predictions== y)
print(f"{accuracy *100}%")
tp = 0
tn = 0
fp = 0
fn = 0
for  actual, predicted in zip(y,predictions):
    if actual == 0 and predicted == 0:
        tn +=1
    elif actual == 1 and predicted == 1:
        tp +=1
    elif actual == 0 and predicted == 1:
        fp +=1
    elif  actual == 1 and predicted == 0:
        fn +=1
confusion_matrix =np.array([[tn,fp],
                           [fn,tp]])
print(confusion_matrix)



mean_plus = 0.5*(mean0+mean1)
mean_minus = (mean0-mean1)
inv_Sigma = np.linalg.inv(Sigma)
decision_boundary = mean_plus @ inv_Sigma @ mean_minus - np.log((1 - phi) / phi)
weights = mean_minus @ inv_Sigma 
x_min = X[:, 0].min()
x_max = X[:, 0].max()
num_steps = 500
x_range = np.linspace(x_min, x_max, num_steps)
w1= weights[0]
w2 = weights[1]
y_range = (decision_boundary - (w1 * x_range)) / w2
plt.plot(x_range,y_range)
plt.show()







   #Ridge regression 
rng =np.random.default_rng()
single_val = rng.uniform()
X =rng.uniform(low=0.0,high=10.0, size=150)
noise_1 = rng.normal(loc=0.0, scale=0.1,size=100)
noise_2 = rng.normal(loc=0.0, scale=0.1,size=50)
X_train = X[:100]
X_test =X[-50:]
y_train = np.sin(X_train)+noise_1
y_test = np.sin(X_test) + noise_2
degree = 1
mean_train = X_train.mean()
std_train = X_train.std()
X_train_scaled=( X_train-mean_train )/std_train
X_test_scaled=(X_test-mean_train )/std_train
train_errors = []
test_errors = []
train_mses = []
test_mses = []
for degree in (1, 3, 15):
    coefficient=np.polyfit(X_train_scaled,y_train,degree)
    y_trainpred=np.polyval(coefficient,X_train_scaled)
    y_testpred=np.polyval(coefficient,X_test_scaled)
    error0=(  y_trainpred-y_train)
    error1=(y_testpred-y_test)
    mean0 = (error0**2).mean()
    train_errors.append(mean0)
    mean1 = (error1**2).mean()
    test_errors.append(mean1)
    mse_new = np.mean((y_train - y_trainpred)**2)
    train_mses.append(mse_new)
    mse_test = np.mean((y_test - y_testpred)**2)
    test_mses.append(mse_test)
print(f" Train MSES: {train_mses}\n")
print(f"Test MSES: {test_mses}\n")
    
X_axis =[1, 3, 15]
y_trainaxis = train_errors
y_testaxis = test_errors
# plt.plot(X_axis,y_trainaxis, color='blue')
# plt.plot(X_axis,y_testaxis,color="red")
# plt.show()
X_design = np.vander(X_train_scaled, 15, increasing=True)
X_test_design = np.vander(X_test_scaled, 15, increasing=True)
X_transpose_X = X_design.T @ X_design
p = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_train_mses = []
ridge_test_mses = []
for i in p:
    I = np.eye(15)
    inverse_mat = (X_transpose_X + i * I)
    theta = np.linalg.inv(inverse_mat) @ X_design.T @ y_train
    predictions = X_design @theta 
    test_predictions = X_test_design @ theta
    Mse = np.mean((y_train -predictions)**2)
    ridge_train_mses.append(Mse)
    Mse_test = np.mean((y_test -test_predictions)**2)
    ridge_test_mses.append(Mse_test)
    print(f"Ridge Train MSE :{Mse}\n")
    print(f"Ridge Test MSE :{Mse_test}\n")
plt.semilogx(p,ridge_test_mses, color='blue')
plt.semilogx(p,ridge_train_mses, color='black')
plt.show()
