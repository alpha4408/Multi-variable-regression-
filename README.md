# Multi-variable-regression-
Python Multi-variable regression 
import pandas as pd
from sklearn import linear_model

text = pd.read_csv('CarPrice.csv')

X = text[['InsuranceRisk', 'Gasoline', 'Turbo', 'NumberDoors', 'CarBodyType', 'Drive', 'WheelBase', 
           'Length', 'Width', 'Height', 'Weight', 'NumCylinders', 'EngineSize', 'Horsepower',
           'PeakRPM', 'CityMPG', 'HighwayMPG']]
y = text['Price']

regr = linear_model.LinearRegression()  #calling the linear regression model
regr.fit(X, y)  #fitting a line to the data 
coeff = regr.coef_  #storing the parameters in coeff

print('\nThe price is determined by: \n')
print('\n(Insurance Risk) * %0.2f.' %coeff[0])
print('\n(Gasoline Type) * %0.2f.' %coeff[1])
print('\n(Type of Intake) * %0.2f.' %coeff[2])
print('\n(Number of Car Doors) * %0.2f.' %coeff[3])
print('\n(Type of Car Body) * %0.2f.' %coeff[4])
print('\n(Type of Drive) * %0.2f.' %coeff[5])
print('\n(Wheel Base) * %0.2f.' %coeff[6])
print('\n(Length of Car) * %0.2f.' %coeff[7])
print('\n(Width of Car) * %0.2f.' %coeff[8])
print('\n(Height of Car) * %0.2f.' %coeff[9])
print('\n(Weight of Car) * %0.2f.' %coeff[10])
print('\n(Number of Cylinders) * %0.2f.' %coeff[11])
print('\n(Engine Size) * %0.2f.' %coeff[12])
print('\n(Horsepower) * %0.2f.' %coeff[13])
print('\n(Peak RPM) * %0.2f.' %coeff[14])
print('\n(City Mileage) * %0.2f.' %coeff[15])
print('\n(Highway Mileage) * %0.2f.' %coeff[16])


print('\n\nThe r-squared value is %0.2f \n' %regr.score(X,y))  #output the r-squared value
