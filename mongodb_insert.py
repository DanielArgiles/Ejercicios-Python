# Módulo insert


from pymongo import MongoClient

# Creando conexiones para comunicar con Mongo DB
Client= MongoClient('localhost:27017')
db=Client.EmployeeData


def insert(): # método o función insert ,para insertar datos a mongo db
	try: 
		employeeid= input('Enter Employee id:')
		employeename= input('Enter Name:')	
		employeeage= input('Enter age:')
		employeecountry= input('Enter Country:')
		
		db.Employees.insert_one( # Insertamos en la base de datos los datos anteriores introducidos por teclado. Estructura JSON.
		{
			"id":employeeid,
			"name": employeename,
			"age":employeeage,
			"country": employeecountry
		})

		print ('Inserted data successfully')

	except ImportError: #Si hay algún error al transferir los datos
		platform_specific_module=None 
		# print str(e)