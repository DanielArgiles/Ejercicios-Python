# Módulo read para leer datos que hay en la base de datos

from pymongo import MongoClient

# Creando conexiones para comunicar con Mongo DB
Client= MongoClient('localhost:27017')
db=Client.EmployeeData


def read(): # método o función read() para leer datos
	try:
		empcol= db.Employees.find() # encontrar datos en la base de datos
		print('\n All data from EmployeeData Database\n')
		for emp in empcol: # itineramos y visualizamos los datos en consola
			print(emp)


	
	except ImportError: #Si hay algún error al transferir los datos
		platform_specific_module=None 
		# print str(e)
