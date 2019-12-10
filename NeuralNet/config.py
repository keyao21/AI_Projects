
DATA_DIR_PATH = './data'
NET_DIR_PATH = './nets'
RESULTS_DIR_PATH = './results'

WDBC_LRATE = 0.1
WDBC_NEPOCH = 100

GRADES_LRATE = 0.05
GRADES_NEPOCH = 100

AUDIT_LRATE = 0.05
AUDIT_NEPOCH = 100

CONFIGS = {
	'WDBC' : {
		'TRAIN': {
			'learning_rate' : WDBC_LRATE,
	        'num_epoch' : WDBC_NEPOCH,
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
			'learning_rate' : GRADES_LRATE,
	        'num_epoch' : GRADES_NEPOCH,
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
			'learning_rate' : WDBC_LRATE,
	        'num_epoch' : 1,
	        'init_network_filename' :'sample.NNWDBC.init',
	        'training_data_filename' : 'wdbc.mini_train',
	        'trained_network_filename' : 'kevin.NNWDBC.{0}.1.mini_trained'.format( str(WDBC_LRATE).split('.')[1], 1 )
		}, 
		'TEST': {
			'trained_network_filename' : 'kevin.NNWDBC.{0}.1.trained'.format( str(WDBC_LRATE).split('.')[1], 1 ),
			'testing_data_filename'  : 'wdbc.test',
			'results_filename' : 'kevin.NNWDBC.{0}.1.results'.format( str(WDBC_LRATE).split('.')[1], 1 )
		}
	},


	# CHANGE THIS STUFF TO LOAD Audit risk NETWORK AND STUFF
	'Audit' : {
		'TRAIN': {
			'learning_rate' : AUDIT_LRATE,
	        'num_epoch' : AUDIT_NEPOCH,
	        'init_network_filename' : 'kevin.NNAudit.init',
	        'training_data_filename' : 'auditrisk.processed.train',
	        'trained_network_filename' : 'kevin.NNAudit.{0}.{1}.trained'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH )
		}, 
		'TEST' : {
			'trained_network_filename' : 'kevin.NNAudit.{0}.{1}.trained'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH ),
	        'testing_data_filename'  : 'auditrisk.processed.test', 
	     	'results_filename' : 'kevin.NNAudit.{0}.{1}.results'.format( str(AUDIT_LRATE).split('.')[1], AUDIT_NEPOCH )   
		}
	}
}