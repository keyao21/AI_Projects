
WDBC_LRATE = 0.1
WDBC_NEPOCH = 100

GRADES_LRATE = 0.05
GRADES_NEPOCH = 100

AUDIT_LRATE = 0.1
AUDIT_NEPOCH = 100

CONFIGS = {
	'WDBC' : {
		'TRAIN': {
			'l_rate' : WDBC_LRATE,
	        'n_epoch' : WDBC_NEPOCH,
	        'init_network_filename' :'sample.NNWDBC.init',
	        'training_data_filename' : 'wdbc.train',
	        'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH )
		}, 
		'TEST': {
			'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH ),
			'testing_data_filename'  : 'wdbc.test',
			'results_filename' : 'kevin.NNWDBC.{0}.{1}.results'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH )
		}
	}, 
	'Grades' : {
		'TRAIN': {
			'l_rate' : GRADES_LRATE,
	        'n_epoch' : GRADES_NEPOCH,
	        'init_network_filename' : 'sample.NNGrades.init',
	        'training_data_filename' : 'grades.train',
	        'trained_network_filename' : 'kevin.NNGrades.{0}.{1}.trained'.format( str(GRADES_LRATE).split('.')[1], GRADES_NEPOCH )
		}, 
		'TEST' : {
			'trained_network_filename' : 'kevin.NNGrades.{0}.{1}.trained'.format( str(GRADES_LRATE).split('.')[1], GRADES_NEPOCH ),
	        'testing_data_filename'  : 'grades.test', 
	     	'results_filename' : 'kevin.NNGrades.{0}.{1}.results'.format( str(GRADES_LRATE).split('.')[1], GRADES_NEPOCH )   
		}
	},

	# CHANGE THIS STUFF TO LOAD Audit risk NETWORK AND STUFF
	'WDBC_mini' : {
		'TRAIN': {
			'l_rate' : WDBC_LRATE,
	        'n_epoch' : WDBC_NEPOCH,
	        'init_network_filename' :'sample.NNWDBC.init',
	        'training_data_filename' : 'wdbc.mini_train',
	        'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.mini_trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH )
		}, 
		'TEST': {
			'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH ),
			'testing_data_filename'  : 'wdbc.test',
			'results_filename' : 'kevin.NNWDBC.{0}.{1}.results'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH )
		}
	},


	# CHANGE THIS STUFF TO LOAD Audit risk NETWORK AND STUFF
	'Audit' : {
		'TRAIN': {
			'l_rate' : AUDIT_LRATE,
	        'n_epoch' : AUDIT_NEPOCH,
	        'init_network_filename' : 'kevin.NNAudit.init',
	        'training_data_filename' : 'auditrisk.train',
	        'trained_network_filename' : 'kevin.NNAudit.{0}.{1}.trained'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH )
		}, 
		'TEST' : {
			'trained_network_filename' : 'kevin.NNAudit.{0}.{1}.trained'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH ),
	        'testing_data_filename'  : 'auditrisk.test', 
	     	'results_filename' : 'kevin.NNAudit.{0}.{1}.results'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH )   
		}
	}


	# # CHANGE THIS STUFF TO LOAD HOUSING NETWORK AND STUFF
	# 'Housing' : {
	# 	'TRAIN': {
	# 		'l_rate' : HOUSING_LRATE,
	#         'n_epoch' : HOUSING_NEPOCH,
	#         'init_network_filename' : 'kevin.NNHousing.init',
	#         'training_data_filename' : 'housing.process.train',
	#         'trained_network_filename' : 'kevin.NNHousing.{0}.{1}.trained'.format( str(HOUSING_LRATE).split('.')[1], HOUSING_NEPOCH )
	# 	}, 
	# 	'TEST' : {
	# 		'trained_network_filename' : 'kevin.NNHousing.{0}.{1}.trained'.format( str(HOUSING_LRATE).split('.')[1], HOUSING_NEPOCH ),
	#         'testing_data_filename'  : 'housing.process.test', 
	#      	'results_filename' : 'kevin.NNHousing.{0}.{1}.results'.format( str(HOUSING_LRATE).split('.')[1], HOUSING_NEPOCH )   
	# 	}
	# }
}