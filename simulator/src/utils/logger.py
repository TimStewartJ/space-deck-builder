def log_to_console(message):
    print(message)

def log_to_file(message, filename='game_log.txt'):
    with open(filename, 'a') as log_file:
        log_file.write(message + '\n')

def log_event(event_type, details):
    message = f"{event_type}: {details}"
    log_to_console(message)
    log_to_file(message)