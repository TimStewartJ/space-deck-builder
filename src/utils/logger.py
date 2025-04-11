from datetime import datetime

verbose = True  # Global flag to control logging
disabled = False  # Global flag to completely disable logging
log_file_name = f"game_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # Global log file name

def set_verbose(value):
    """
    Set the verbose flag to control logging output.
    
    Args:
        value (bool): Whether to enable verbose logging
    """
    global verbose
    verbose = value

def set_disabled(value):
    """
    Set the disabled flag to completely disable logging.
    
    Args:
        value (bool): Whether to disable all logging
    """
    global disabled
    disabled = value

def log(message):
    """
    Log a message to the console and a file.
    
    Args:
        message (str): The message to log.
    """
    if not disabled:
        log_to_console(message)
        log_to_file(message)

def log_verbose(message):
    """
    Log a message if verbose mode is enabled.
    
    Args:
        message (str): The message to log.
    """
    if not disabled and verbose:
        log_to_console(message)
        log_to_file(message)

def log_to_console(message):
    if not disabled and verbose:
        print(message)

def log_to_file(message, filename=log_file_name):
    if not disabled and verbose:
        with open(filename, 'a') as log_file:
            log_file.write(message + '\n')

def log_event(event_type, details):
    if not disabled and verbose:
        message = f"{event_type}: {details}"
        log_to_console(message)
        log_to_file(message)