
WDBC_LRATE = 0.1
WDBC_NEPOCH = 200

GRADES_LRATE = 0.1
GRADES_NEPOCH = 200

CONFIGS = {
	'WDBC' : {
		'TRAIN': {
			'l_rate' : WDBC_LRATE,
	        'n_epoch' : WDBC_NEPOCH,
	        'init_network_filename' :'kevin.NNWDBC.init',
	        'training_data_filename' : 'wdbc.train',
	        'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH )
		}, 
		'TEST': {
			'trained_network_filename' : 'kevin.NNWDBC.{0}.{1}.trained'.format( str(WDBC_LRATE).split('.')[1], WDBC_NEPOCH ),
			'testing_data_filename'  : 'wdbc.test'
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
	        'testing_data_filename'  : 'grades.test'
		}
	}
}