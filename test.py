def a(x):
    l=0
    m=1
    def b(x):
        print(l)
        print(m)
        print(x)
        m=5
    b(x)

def c(x):
    a(x)

c(4)