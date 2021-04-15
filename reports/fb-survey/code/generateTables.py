import pandas as pd

def addAreaTable(engine):
	"""
		generate a SQL table named area in the SQLite engine
		
		Arguments:
			engine(sqlite obj): the object returned by sqlalchemy.create_engine()
		Return:
			None
	"""

	# generate float numbers from 1 to 53
	key = [float(i) for i in range(1,54)]
	
	# build the list of areas that follow survey design
	area_ls = ['Alabama', 'Alaska','Arizona','Arkansas', 'California' , 'Colorado', 'Connecticut',\
           "Delaware", "District of Columbia","Florida", "Georgia", "Hawaii", "Idaho", "Illinois", \
           "Indiana" , "Iowa", "Kansas", "Kentucky" ,"Louisiana", 'Maine' , 'Maryland',\
           'Massachusetts',  'Michigan', 'Minnesota','Mississippi' ,'Missouri' , 'Montana',\
           'Nebraska',  'Nevada' , 'New Hampshire',"New Jersey", "New Mexico", "New York", \
           "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon" ,"Pennsylvania", \
           "Puerto Rico", "Rhode Island", "South Carolina","South Dakota", "Tennessee", \
           "Texas" , "Utah", "Vermont" , "Virginia" , "Washington", "West Virginia" , \
           "Wisconsin" , "Wyoming", "Not in US"]

	d = {'state_id': key, 'state_name': area_ls}
	df = pd.DataFrame(data=d)
	df.to_sql("area", engine, if_exists='replace', index= False)
	
	return None

		
def addJobTable(engine):
	"""
                generate a SQL table named job in the SQLite engine
                
                Arguments:
                        engine(sqlite obj): the object returned by sqlalchemy.create_engine()
                Return:
                        None
	"""
	# generate float numbers from 1 to 16
	key = [float(i) for i in range(1,17)]
	# build the list of areas that follow survey design
	job_ls = ["Community and social service", "Education, training, and library", "Arts, design, entertainment, sports, and media", "Healthcare practitioners and technicians", "Healthcare support", "Protective service", "Food preparation and serving related", "Building and grounds cleaning and maintenance", "Personal care and service (not healthcare)", "Sales and related", "Office and administrative support", "Construction and extraction", "Installation, maintenance, and repair", "Production", "Transportation and material moving", "Other occupation"]
	d = {'job_id': key, 'job_name': job_ls}
	df = pd.DataFrame(data=d)
	df.to_sql("job", engine, if_exists='replace', index= False)
	return None
