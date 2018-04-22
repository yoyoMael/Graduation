class A:
num1 = None
def __init__(self, num1):
    A.num1 = num1

class B:
num2 = None
def __init__(self, num2):
    B.num2 = num2

class C(A,B):

def __init__(self):
    pass

def num(self):
    return B.num2 + A.num1

if __name__ == '__main__':
a = A(2)
b = B(5)
c = C()
print "C num1"
print C.num1
print "C num2"
print C.num2
print "Summe"
print c.num()