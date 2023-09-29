import math

lam, mu, x, y, a, k = 1, 1, .9, .9, 4, 2

def tnext(t,l,x,m,y,a,k):
    return math.log (1 + k * math.pow (a, 1 / (1 - math.exp(-m*t)))) / (l/(1-x) + m/(1-y))
#    rho = max(l/(1-x),m/(1-y))
#    return math.log(1 + k * math.pow (a, 1 / (1 - math.exp(-(l+m)*t)))) / rho

t = max((1-x)/lam,(1-y)/mu) * math.log (1+a)
for i in range(100):
    print (i, t)
    t = tnext(t,lam,x,mu,y,a,k)
