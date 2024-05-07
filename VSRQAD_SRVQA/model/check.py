
class A():
    def initialize(self):
        self.printa()

    def printa(self):
        print('a')


class B(A):
    def initialize(self):
        A.initialize(self)

    def printa(self):
        print('b')



if __name__ == '__main__':
    b = B()
    b.initialize()