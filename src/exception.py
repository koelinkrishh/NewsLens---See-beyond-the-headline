## Exception Info setup
import sys

def error_details(error, error_detail:sys):
    """
    Function for getting detailed error message
    
    Parameters:
        error (Exception): The exception object containing the error message.
        error_detail (sys): The sys module containing the exception details.
    
    Returns:
        str: detailed error message with file name and line number
    """
    
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f""" Error occured in [{filename}], lineno: [{exc_tb.tb_lineno}].
      -> error: [{str(error)}] """
      
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_msg_detail:sys):
        """
        Initializes a CustomException object with the given error message and error details.

        Parameters:
            error_message (str): The error message to be displayed.
            error_msg_detail (sys): The sys module containing the exception details.
        """
        super().__init__(error_message)
        self.error_msg = error_details(error_message, error_detail=error_msg_detail)

    def __str__(self):
        """ Return the error message string representation of the exception. """
        return self.error_msg