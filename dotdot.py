import time
import sys

def print_with_ellipsis(text, delay=0.5):
    """
    Print text immediately followed by an ellipsis with a delay.

    Parameters:
    text (str): The text to be printed.
    delay (float): The delay before printing the ellipsis.
    """
    # Print the main text immediately
    print(text, end='', flush=True)
    
    # Delay before printing the ellipsis
    time.sleep(delay)
    
    # Print the ellipsis
    sys.stdout.write(" ...")
    sys.stdout.flush()


