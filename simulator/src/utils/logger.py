verbose = True  # Global flag to control logging

def set_verbose(value):
    """
    Set the verbose flag to control logging output.
    
    Args:
        value (bool): Whether to enable verbose logging
    """
    global verbose
    verbose = value

def log(message):
    """
    Log a message to the console and a file.
    
    Args:
        message (str): The message to log.
    """
    log_to_console(message)
    log_to_file(message)

def log_verbose(message):
    """
    Log a message if verbose mode is enabled.
    
    Args:
        message (str): The message to log.
    """
    if verbose:
        log_to_console(message)
        log_to_file(message)

def log_to_console(message):
    if verbose:
        print(message)

def log_to_file(message, filename='game_log.txt'):
    if verbose:
        with open(filename, 'a') as log_file:
            log_file.write(message + '\n')

def log_event(event_type, details):
    if verbose:
        message = f"{event_type}: {details}"
        log_to_console(message)
        log_to_file(message)