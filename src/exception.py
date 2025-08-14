import sys
from src.logger import logging

class Airlines_Exeption(Exception):
    def __init__(self, error_message,error_details:sys):
        logging.info("Defining Custom Exception")
        self.error_message = error_message
        _,_,exc_tb = sys.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_no = exc_tb.tb_lineno
    
    def __str__(self):
        return "Error occured in python script [{0}] line number [{1}] error message [{2}]".format(self.file_name,self.line_no,str[self.error_message])