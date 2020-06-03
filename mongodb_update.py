# Módulo update

from pymongo import MongoClient

# Creando conexiones para comunicar con Mongo DB
Client= MongoClient('localhost:27017')
db=Client.EmployeeData


def update(): # método o función update() para actualizar datos
	try:
		criteria= input('\n Enter id to update\n')
		name= input('\n Enter name to updaten')
		age= input('\n Enter age to update\n')
		country= input('\n Enter country to update\n')


		db.Employees.update_one(
		{"id": criteria}, # para poder modificar un campo en la base de datos separamos el id
		{
			"$set": { # modificamos con el set los datos de name, age, country
				 "name":name,
				 "age":age,
				 "country":country
			}
		}
		)
		print("\nRecords updated successfully\n")


	except ImportError: #Si hay algún error al transferir los datos
		platform_specific_module=None 
		# print str(e)
