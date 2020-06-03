# Módulo delete

from pymongo import MongoClient

# Creando conexiones para comunicar con Mongo DB
Client= MongoClient('localhost:27017')
db=Client.EmployeeData

def delete(): # método o función delete() para actualizar datos
	try:
		criteria= input('\n Enter employee id to delete\n') # introducimos el id que queremos eliminar
		db.Employees.delete_many({"id":criteria}) # con esta consulta eliminamos
		print('\n Deletion successful\n')
	
	except ImportError: #Si hay algún error al transferir los datos
		platform_specific_module=None 
		# print str(e)