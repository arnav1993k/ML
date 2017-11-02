#!/usr/bin/env python3
import os
from Utilities import Utilities
def main():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	training_path=desktop_path+"/trainingDigits"
	testing_path=desktop_path+"/testDigits"
	util=Utilities()
	features,outputs=util.converttoVector(training_path)
	util.getImages(features,32,32,outputs)
#main
if __name__ == '__main__':
	main()