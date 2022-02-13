# Python Basics

  * Strings, integers, floats, functions, conditionals

### Useful functions

  * print() // prints a value
  * type() // describe the type of thing

  * Operator	Name	Description
      a + b	Addition	Sum of a and b
      a - b	Subtraction	Difference of a and b
      a * b	Multiplication	Product of a and b
      a / b	True division	Quotient of a and b
      a // b	Floor division	Quotient of a and b (removing fractional parts)
      a % b	Modulus	Integer remainder after division of a by b
      a ** b	Exponentiation	a raised to the power of b
      -a	Negation	The negative of a

  * Order of operations
      PEMDAS
        - Parentheses
        - Exponents
        - Multiplication
        - Division
        - Addition
        - Subtraction

  * Built-in functions for working with numbers
    - min(1,2,3) = 1
    - max(1,2,3) = 3
    - abs(32) / abs(-32) = 32
    - convert their arguments to the corresponding type
      - float(10) = 10.0
      - int("807") = 807

## Defining functions

def least_difference(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)
