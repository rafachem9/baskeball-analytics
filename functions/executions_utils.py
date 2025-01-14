import time


def execution(func, process_name):
    # Mark the start time
    start_time = time.time()
    print(f"Starting {process_name}")

    # Execute the function
    func()

    # Mark the end time
    end_time = time.time()

    # Calculate and display the duration
    duration = end_time - start_time
    print(f"Execution time: {round(duration, 1)} seconds")
