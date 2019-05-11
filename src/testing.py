import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print ( sigmoid((sigmoid(.54) *.35)  +  (sigmoid(.51) *.4)  + .25) )

a =  ((.15 - 0.7194006710513193) * (.15 - 0.7194006710513193) )/2
b = ((.85 - 0.6728397191051313) * (.85 - 0.6728397191051313) )/2

print (a+b)