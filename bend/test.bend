def fib(n):
  switch n:
    case 0:
      return 1
    case 1:
      return 1
    case _:
      return fib(n - 1)+fib(n - 2)

def main:
  return fib(30)


