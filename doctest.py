'''
Doctest es un framework que viene en Python el cual permite desarrollar aplicaciones utilizando TDD (Desarrollo guiado por pruebas).
El TDD exige escribir las pruebas primero y la refactorización del código para llegar al resultado deseado.
En este caso se usará doctest el cual permite realizar pruebas según la documentación que se tenga escrita en el código.
Significa que es necesario tener una documentación clara para cada función antes de desarrollarla, de esta forma se tiene claro los casos de funcionamiento correcto de la función y los casos en los cuales puede fallar.

# Ejecutamos el scrpt desde cmd:
# python test.py : si no hay fallos, no se muestra nada
# python test.py -v : mostramos todo el registro de ejecución, con el resultado de los tests según lo esperado en la documentación.
'''


# Ejercicio 1

def suma(a, b):
	"""Docstring: Esta función recibe dos parámetros y devuelve la suma de ambos

	>>> suma(5,10)
	15 # Lo que esperamos, si coincide con el test, obtendremos un OK

	>>> suma(-5,7)
	2

    Cadenas de texto:

    >>> suma('aa','bb')
    'aabb'

    O listas:

    >>> a = [1, 2, 3]
    >>> b = [4, 5, 6]
    >>> suma(a,b)
    [1, 2, 3, 4, 5, 6]

    Sin embargo no podemos sumar elementos de tipos diferentes:

    >>> suma(10,"hola")
    "10hola" # Eliminamos esta línea que esperábamos, al añadir las de abajo Traceback obtenido tras realizar una primera prueba...

    Traceback (most recent call last):
      ...
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
	"""
	return a+b

if __name__ == '__main__':
    import doctest
    doctest.testmod()
# Ejercicio 2

def palindromo(palabra):
    """
    Comprueba si una palabra es un palíndrimo. Los palíndromos son 
    palabras o frases que se leen igual de ambos lados.
    Si es un palíndromo devuelve True y si no False

    >>> palindromo("radar")
    True

    >>> palindromo("somos")
    True

    >>> palindromo("holah")
    False

    >>> palindromo("Atar a la rata")
    True
    """
    # A continuación arreglamos los errores debio a diferencias de espacios y minúsculas al leer de izda a decha y viceversa que encontramos en "Atar a la rata"
    if palabra.lower().replace(" ", "") == palabra[::-1].lower().replace(" ", ""):  # [::-1] obtiene una lista "del revés"
        return True
    else:
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# Ejercicio 3

def doblar(lista):
    """Dobla el valor de los elementos de una lista
    >>> l = [1, 2, 3, 4, 5] 
    >>> doblar(l)
    [2, 4, 6, 8, 10]

    >>> l = [] 
    >>> for i in range(10):
    ...     l.append(i)
    >>> doblar(l)
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    """
    return [n*2 for n in lista]

if __name__ == '__main__':
    import doctest
    doctest.testmod()

