import subprocess
import sys
import time

default_msg = \
"""Usage: python run.py <dataset> <task_num=5> ; executes tasks up to task_num on specified dataset
<dataset>: antique, neurips
<task_num>: 1, 2, 3, 4, 5 ; default: 5"""

def main():
    if len(sys.argv) < 2:
        print(default_msg)
        return
    
    dataset = sys.argv[1]
    task_num = 5
    if len(sys.argv) > 2:
        task_num = int(sys.argv[2])
    
    time_start = time.perf_counter()

    if dataset == 'antique':
        from task_antique import main as task_main
    elif dataset == 'neurips':
        from task_neurips import main as task_main
    else:
        print(default_msg)
        return

    print("Running...")
    task_main(task_num)
    
    print(f"Total time elapsed: {time.perf_counter() - time_start}")

if __name__ == "__main__":
    main()