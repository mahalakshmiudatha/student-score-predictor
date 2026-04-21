import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv("data.csv")
x=data[['hours_studied','sleep_hours','previous_score']]
y=data['final_score']
model=LinearRegression()
model.fit(x,y)
prediction=model.predict([[4,7,65]])
print("predicted score:",prediction[0])