print("Hello Python")
print(3+2)
# https://www.youtube.com/watch?v=9DLWngQ4M48
# Tensorflow를 위한 설정을 성공적으로 수행했음
# I installed tensorflow.

# define function!!!
def my_main() :
    print("Hello my_main function")

def odd_even(input_number) :
    if input_number % 2 == 0 :
        print("even number")
    else :
        print("Odd number")

class MyClass:
    def __init__(self, _a):
        self.a = _a
        print(self.a)

if __name__ == '__main__' :
    mc = MyClass(3)
    my_main()
    odd_even(1)
    odd_even(2)

    # for loop
    for i in range(0,10) :
        my_main()

   
