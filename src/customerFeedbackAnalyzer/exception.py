import sys
from src.customerFeedbackAnalyzer.logger import logging

def error_message_detail(error,error_detail:sys):
    """
    This function generates a detailed error message based on the provided error and error details.

    Parameters:
        error (object): The error object that occurred.
        error_detail (sys): The error details, including the exception information.

    Returns:
        str: A formatted error message containing the script name, line number, and error message.
    """
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        """
        Initializes a new instance of the CustomException class.

        Parameters:
            error_message (object): The error object that occurred.
            error_details (sys): The error details, including the exception information.

        Returns:
            None
        """
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_details)

    def __str__(self):
        """
        Returns a string representation of the CustomException instance.
        
        This method overrides the default string representation of the Exception class to provide a more informative error message.
        
        Returns:
            str: A string representation of the CustomException instance, containing the error message.
        """
        return self.error_message