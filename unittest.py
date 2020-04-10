'''
Unit testing: Pruebas en el código para comprobar su correcto funcionamiento.

Respuestas de las pruebas: 
OK: Para indicar que la prueba se ha pasado exitosamente.
FAIL (F): Para indicar que la prueba no ha pasado éxitosamente lanzaremos una excepción AssertionError (sentencia verdadero-falso)
ERROR (E): Para indicar que la prueba no ha pasado éxitosamente, pero el resultado en lugar de ser una aserción es otro error.
'''


import unittest

# Ejercicio 1: Prueba unitaria sencilla

class Pruebas (unittest.TestCase):
    def test(self):
    # Selecccionamos una de las 3 opciones y ejecutamos el código:
    	pass # La prueba devuelve OK
       # raise AssertionError() # La prueba devuelve F al invocar a una excepción
       #1/0 # La prueba devuelve E
    

#if __name__ == "__main__":  
 #   unittest.main()        # cierra el import unittest, como tenemos varios ejercicios en el mismo script , dejaré solo el del final


# Ejercicio 2: Funciones propias: 3 tests OK

def doblar(a): return a*2
def sumar(a,b): return a+b  
def es_par(a): return 1 if a%2 == 0 else 0

class PruebasFunciones(unittest.TestCase):
	# assertEqual (a,b) comprueba que a == b
    def test_doblar(self):
        self.assertEqual(doblar(10), 20) #  doblar 10 es igual a 20
        self.assertEqual(doblar('Ab'), 'AbAb') # doblar Ab es igual a Abab

    def test_sumar(self):
        self.assertEqual(sumar(-15, 15), 0) # sumar 15 y -15 es igual a 0
        self.assertEqual(sumar('Ab' ,'cd'), 'Abcd') #sumar Ab + cd es igual a Abcd

    def test_es_par(self):
        self.assertEqual(es_par(11), False) # El núumero 11 no es par, por tanto devuelve False
        self.assertEqual(es_par(68), True) # El número 68 es par, por tanto devuelve True


#if __name__ == '__main__':
 #   unittest.main()


# Ejercicio 3: métodos de cadenas: 3 tests OK

class PruebasMetodosCadenas(unittest.TestCase):
	# assertEqual (a,b) comprueba que a == b
	# assertTrue(x) comprueba que bool(x) es True
	# assertFalse(x) comprueba que bool(x) es False

    def test_upper(self):
        self.assertEqual('hola'.upper(), 'HOLA') # hola en mayúsculas es HOLA

    def test_isupper(self):
        self.assertTrue('HOLA'.isupper()) # HOLA.isuper() devuelve True porque está en mayúsculas
        self.assertFalse('Hola'.isupper()) # Hola.isuper() devuelve False porque no todas las letras son mayúsculas

    def test_split(self):
        s = 'Hola mundo'
        self.assertEqual(s.split(), ['Hola', 'mundo']) # split() devuelve una lista de cadenas con las palabras separadas por una coma


#if __name__ == '__main__':
 #   unittest.main()


# Ejercicio 4: Preparación y limpieza :  1 test OK

# La clase TestCase incorpora dos métodos: 
# setUp():  prepara el contexto de las pruebas,por ejemplo, escribir unos valores de prueba en un fichero, conectarse a un servidor o a una base de datos.
# tearDown(): hace lo propio con la limpieza, borrar el fichero, desconectarse del servidor o borrar las entradas de prueba de la base de datos.

def doblar(a): return a*2

class PruebaTestFixture(unittest.TestCase):
	# assertEqual (a,b) comprueba que a == b
    def setUp(self):
        print("Preparando el contexto")
        self.numeros = [1, 2, 3, 4, 5]

    def test(self):
        print("Realizando una prueba")
        r = [doblar(n) for n in self.numeros]
        self.assertEqual(r, [2, 4, 6, 8, 10]) # r es una lista de números doblados

    def tearDown(self):
        print("Destruyendo el contexto")
        del(self.numeros)


if __name__ == '__main__':
    unittest.main() 



